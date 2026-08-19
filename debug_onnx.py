"""
Script debug: kiểm tra model ONNX có bị lỗi cấu hình không.
Chạy: python3 debug_onnx.py
"""

import os
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

MODEL_DIR = "model_onnx"

print("=" * 70)
print("1. KIỂM TRA FILE TRONG THƯ MỤC MODEL")
print("=" * 70)
for f in sorted(os.listdir(MODEL_DIR)):
    print(" -", f)

print()
print("=" * 70)
print("2. LOAD TOKENIZER + MODEL")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)

print("pad_token_id:", tokenizer.pad_token_id)
print("eos_token_id:", tokenizer.eos_token_id)
print("model.config.decoder_start_token_id:",
      getattr(model.config, "decoder_start_token_id", "KHÔNG CÓ"))

if hasattr(model, "generation_config"):
    print("generation_config.decoder_start_token_id:",
          getattr(model.generation_config, "decoder_start_token_id", "KHÔNG CÓ"))
else:
    print("!!! model KHÔNG có generation_config -> đây có thể là nguyên nhân lỗi")

print()
print("=" * 70)
print("3. TEST GENERATE - PHIÊN BẢN TỐI GIẢN (không dùng repetition_penalty)")
print("=" * 70)

text = "check the document before the meeting"
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

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)

# Thử bản đơn giản nhất trước — chỉ greedy, không thêm tham số phụ
output_ids = model.generate(
    **inputs,
    max_new_tokens=96
)

raw = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("OUTPUT (tối giản):", repr(raw))

print()
print("=" * 70)
print("4. TEST GENERATE - GIỐNG HỆT app_onnx.py (đầy đủ tham số)")
print("=" * 70)

output_ids2 = model.generate(
    **inputs,
    max_new_tokens=96,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3
)

raw2 = tokenizer.decode(output_ids2[0], skip_special_tokens=True)
print("OUTPUT (đầy đủ tham số):", repr(raw2))