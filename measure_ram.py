"""
Đo RAM thực tế (RSS) của chính tiến trình Python đang chạy.

Cách dùng:
1. Cài psutil:  pip install psutil
2. Chạy file này TRƯỚC, nó sẽ tự load model + encoder/decoder
   giống hệt app_onnx_manual.py, rồi in RAM tại từng mốc.
3. So sánh với giới hạn 512 MB của Render free tier.
"""

import os
import psutil

process = psutil.Process(os.getpid())


def print_ram(label):
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[RAM] {label}: {mem_mb:.1f} MB")


print_ram("Trước khi import bất kỳ thư viện nào")

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

print_ram("Sau khi import onnxruntime + transformers")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_onnx_int8")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

print_ram("Sau khi load tokenizer")

providers = ["CPUExecutionProvider"]

encoder_session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "encoder_model.onnx"),
    providers=providers
)

print_ram("Sau khi load encoder_model.onnx")

decoder_session = ort.InferenceSession(
    os.path.join(MODEL_DIR, "decoder_model.onnx"),
    providers=providers
)

print_ram("Sau khi load decoder_model.onnx (SẴN SÀNG NHẬN REQUEST)")

# --------------------------------------------------------
# Giả lập 1 request thật để đo RAM lúc PEAK (lúc generate)
# --------------------------------------------------------

text = "Please check all the document for the meeting with ms MBS, done it before 3PM"

prompt = (
    "Convert the following English chat message "
    "into exactly one task.\n\n"
    "Rules:\n"
    "1. Create a short and clear task title.\n"
    "2. Notes should contain the task details.\n"
    "3. Do NOT include deadlines or due times in notes.\n"
    "4. Do NOT create multiple tasks.\n"
    "5. Return ONLY valid JSON.\n"
    "6. Use exactly this format:\n"
    '{"tasks":[{"title":"...","notes":"..."}]}\n\n'
    "Chat:\n"
    f"{text}\n\n"
    "Output:\n"
)

encoded = tokenizer(prompt, return_tensors="np", truncation=True, max_length=512)
input_ids = encoded["input_ids"].astype(np.int64)
attention_mask = encoded["attention_mask"].astype(np.int64)

encoder_outputs = encoder_session.run(
    None,
    {"input_ids": input_ids, "attention_mask": attention_mask}
)

print_ram("Sau khi chạy xong ENCODER 1 request")

encoder_hidden_states = encoder_outputs[0]

decoder_input_names = [x.name for x in decoder_session.get_inputs()]

generated = [tokenizer.pad_token_id or 0]

for step in range(96):

    decoder_input_ids = np.array([generated], dtype=np.int64)

    decoder_inputs = {}
    for name in decoder_input_names:
        if name in ("decoder_input_ids", "decoder_input_ids_0", "input_ids"):
            decoder_inputs[name] = decoder_input_ids
        elif name in ("encoder_hidden_states", "encoder_hidden_states_0"):
            decoder_inputs[name] = encoder_hidden_states
        elif name == "encoder_attention_mask":
            decoder_inputs[name] = attention_mask
        elif name == "attention_mask":
            decoder_inputs[name] = np.ones_like(decoder_input_ids)

    outputs = decoder_session.run(None, decoder_inputs)
    next_token_id = int(np.argmax(outputs[0][:, -1, :], axis=-1)[0])
    generated.append(next_token_id)

    if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
        break

print_ram("Sau khi chạy xong DECODER (generate xong 1 câu trả lời) - PEAK RAM")

print()
print("=" * 60)
print("KẾT LUẬN:")
print("Nếu RAM 'PEAK' ở trên < ~400 MB -> AN TOÀN cho Render free (512MB)")
print("Nếu RAM 'PEAK' > ~450 MB -> RỦI RO, có thể bị OOM khi deploy")
print("=" * 60)