import os
import json
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model_onnx_int8")

ENCODER_PATH = os.path.join(MODEL_DIR, "encoder_model.onnx")
DECODER_PATH = os.path.join(MODEL_DIR, "decoder_model.onnx")


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy encoder model: {ENCODER_PATH}"
    )

if not os.path.exists(DECODER_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy decoder model: {DECODER_PATH}"
    )


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)


# ============================================================
# ONNX SESSIONS
# ============================================================

providers = ["CPUExecutionProvider"]

encoder_session = ort.InferenceSession(
    ENCODER_PATH,
    providers=providers
)

decoder_session = ort.InferenceSession(
    DECODER_PATH,
    providers=providers
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Task Extractor ONNX API",
    version="1.0.0"
)


# ============================================================
# REQUEST
# ============================================================

class PredictRequest(BaseModel):
    text: str


# ============================================================
# HELPER
# ============================================================

def get_input_names(session):
    return [x.name for x in session.get_inputs()]


def get_output_names(session):
    return [x.name for x in session.get_outputs()]


# ============================================================
# MODEL INFO
# ============================================================

print("=" * 70)
print("TASK EXTRACTOR ONNX")
print("=" * 70)

print("Model directory:", MODEL_DIR)

print("\nEncoder inputs:")
for x in encoder_session.get_inputs():
    print(" ", x.name, x.shape, x.type)

print("\nEncoder outputs:")
for x in encoder_session.get_outputs():
    print(" ", x.name, x.shape, x.type)

print("\nDecoder inputs:")
for x in decoder_session.get_inputs():
    print(" ", x.name, x.shape, x.type)

print("\nDecoder outputs:")
for x in decoder_session.get_outputs():
    print(" ", x.name, x.shape, x.type)

print("=" * 70)


# ============================================================
# PROMPT
# ============================================================
#
# QUAN TRỌNG:
#
# Model được fine-tune để phản hồi đúng khi nhận PROMPT ĐẦY ĐỦ
# (rules + format JSON yêu cầu), giống hệt lúc train.
#
# Nếu chỉ đưa text gốc vào thẳng tokenizer (thiếu bước này),
# model sẽ không biết cần sinh JSON, dẫn đến việc nó chỉ
# "copy" lại nguyên văn câu input.
#
# ============================================================

def build_prompt(text):

    return (
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


# ============================================================
# GENERATION
# ============================================================

def generate_text(
    text,
    max_new_tokens=128,
    num_beams=1,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3
):
    """
    Generate text using encoder_model.onnx + decoder_model.onnx.

    This implementation is designed for the exported T5/FLAN-T5
    ONNX structure in model_onnx.

    ------------------------------------------------------------
    QUAN TRỌNG - LỖI ĐÃ SỬA:

    Decoder ONNX (export riêng, không dùng optimum wrapper)
    đặt tên input token CỦA NÓ là "input_ids" - nhưng đây LÀ
    chuỗi decoder_input_ids đang được sinh dần (bắt đầu từ
    decoder_start_token_id), KHÔNG PHẢI input_ids của câu gốc
    đưa vào encoder.

    Bản cũ gán nhầm input_ids gốc vào decoder mỗi vòng lặp,
    khiến decoder không bao giờ thấy chuỗi mình đang sinh ra
    -> luôn dự đoán cùng 1 token -> output toàn dấu chấm lặp lại.
    ------------------------------------------------------------
    """

    # --------------------------------------------------------
    # TOKENIZE INPUT (ĐÃ BỌC PROMPT)
    # --------------------------------------------------------

    prompt = build_prompt(text)

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=512
    )

    input_ids = encoded["input_ids"].astype(np.int64)

    attention_mask = encoded["attention_mask"].astype(np.int64)

    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    encoder_inputs = {}

    encoder_input_names = get_input_names(encoder_session)

    if "input_ids" in encoder_input_names:
        encoder_inputs["input_ids"] = input_ids

    if "attention_mask" in encoder_input_names:
        encoder_inputs["attention_mask"] = attention_mask

    encoder_outputs = encoder_session.run(
        None,
        encoder_inputs
    )

    # First encoder output is normally last_hidden_state
    encoder_hidden_states = encoder_outputs[0]

    # --------------------------------------------------------
    # DECODER INPUT NAMES
    # --------------------------------------------------------

    decoder_input_names = get_input_names(decoder_session)

    # --------------------------------------------------------
    # START TOKEN
    # --------------------------------------------------------

    decoder_start_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    if decoder_start_token_id is None:
        decoder_start_token_id = 0

    generated = [
        decoder_start_token_id
    ]

    # --------------------------------------------------------
    # GREEDY DECODING
    # --------------------------------------------------------

    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):

        decoder_input_ids = np.array(
            [generated],
            dtype=np.int64
        )

        decoder_inputs = {}

        for name in decoder_input_names:

            if name in (
                "decoder_input_ids",
                "decoder_input_ids_0"
            ):
                decoder_inputs[name] = decoder_input_ids

            elif name in (
                "encoder_hidden_states",
                "encoder_hidden_states_0"
            ):
                decoder_inputs[name] = encoder_hidden_states

            elif name == "encoder_attention_mask":
                decoder_inputs[name] = attention_mask

            # FIX: "input_ids" của decoder graph = chuỗi đang
            # sinh dần (decoder_input_ids), KHÔNG PHẢI input_ids
            # của câu gốc.
            elif name == "input_ids":
                decoder_inputs[name] = decoder_input_ids

            # FIX: nếu decoder có input "attention_mask" riêng,
            # đó là mask CỦA decoder_input_ids (toàn số 1 vì
            # không có padding khi generate từng bước một),
            # không phải attention_mask của câu gốc/encoder.
            elif name == "attention_mask":
                decoder_inputs[name] = np.ones_like(decoder_input_ids)

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        outputs = decoder_session.run(
            None,
            decoder_inputs
        )

        logits = outputs[0]

        # Last generated position
        next_token_logits = logits[:, -1, :].copy()

        # --------------------------------------------------
        # REPETITION PENALTY
        #
        # Giảm xác suất các token ĐÃ xuất hiện trong `generated`,
        # giống hệt tham số repetition_penalty trong
        # transformers.generate() của bản torch gốc.
        # --------------------------------------------------

        if repetition_penalty and repetition_penalty != 1.0:

            for token_id in set(generated):

                score = next_token_logits[0, token_id]

                if score > 0:
                    next_token_logits[0, token_id] = score / repetition_penalty
                else:
                    next_token_logits[0, token_id] = score * repetition_penalty

        # --------------------------------------------------
        # NO REPEAT NGRAM
        #
        # Chặn hoàn toàn các token khiến n-gram cuối bị lặp lại
        # (giống no_repeat_ngram_size trong bản torch gốc).
        # --------------------------------------------------

        if no_repeat_ngram_size and len(generated) >= no_repeat_ngram_size:

            n = no_repeat_ngram_size

            prev_ngrams = {}

            for i in range(len(generated) - n + 1):

                ngram = tuple(generated[i:i + n - 1])
                next_tok = generated[i + n - 1]

                prev_ngrams.setdefault(ngram, set()).add(next_tok)

            current_prefix = tuple(generated[-(n - 1):])

            banned_tokens = prev_ngrams.get(current_prefix, set())

            for token_id in banned_tokens:

                next_token_logits[0, token_id] = -1e9

        next_token_id = int(
            np.argmax(
                next_token_logits,
                axis=-1
            )[0]
        )

        generated.append(next_token_id)

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if eos_token_id is not None:
            if next_token_id == eos_token_id:
                break

    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    output_ids = np.array(
        [generated],
        dtype=np.int64
    )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return result.strip()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "task_extractor_v3_flan_t5_small",
        "backend": "ONNX Runtime"
    }


# ============================================================
# JSON PARSER
# ============================================================
#
# Parse output thô của model thành {"tasks":[{"title","notes"}]}
# — GIỮ NGUYÊN logic robust parser từ app.py gốc, để khớp với
# format mà HuggingFaceService.gs bên Apps Script đang mong đợi.
#
# ============================================================

def parse_model_output(raw_output: str):

    import re as _re

    if raw_output is None:
        raw_output = ""

    raw = str(raw_output).strip()

    raw = raw.replace("<pad>", "")
    raw = raw.replace("</s>", "")
    raw = raw.replace("<s>", "")
    raw = raw.strip()

    raw = (
        raw.replace("\u201c", '"')
           .replace("\u201d", '"')
           .replace("\u2018", "'")
           .replace("\u2019", "'")
    )

    try:
        data = json.loads(raw)

        if isinstance(data, dict):
            tasks = data.get("tasks")
            if isinstance(tasks, list) and len(tasks) > 0:
                first_task = tasks[0]
                if isinstance(first_task, dict):
                    title = str(first_task.get("title", "")).strip()
                    notes = str(first_task.get("notes", "")).strip()
                    if title and notes:
                        return {"tasks": [{"title": title, "notes": notes}]}
    except Exception:
        pass

    title = ""
    title_patterns = [
        r'"title"\s*:\s*"([^"]+)"',
        r'"title"\s*"[^"]*"\s*:\s*"([^"]+)"',
        r'title\s*:\s*"([^"]+)"',
    ]
    for pattern in title_patterns:
        match = _re.search(pattern, raw, flags=_re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            if title:
                break

    notes = ""
    notes_match = _re.search(r'"notes"', raw, flags=_re.IGNORECASE)
    if notes_match:
        notes_part = raw[notes_match.end():]
        notes_part = _re.sub(
            r'^\s*(?:[:\-]?\s*(?:Optimize|optimized|optimization)(?:\s+to)?)?\s*[:\-]?\s*',
            '', notes_part, flags=_re.IGNORECASE
        )
        notes_part = notes_part.lstrip()

        # FIX: bỏ TẤT CẢ ký tự rác ở đầu (dấu phẩy, ngoặc kép,
        # hai chấm, khoảng trắng...) thay vì chỉ 1 dấu ngoặc kép
        # như trước — vì model đôi khi sinh ra nhiều ký tự thừa
        # liên tiếp, ví dụ: ,"Check the documentation.
        notes_part = _re.sub(r'^[\s:,"\'\-]+', '', notes_part)

        notes_part = notes_part.strip()

        notes_part = _re.sub(r'"\s*\]?\s*\}?\s*$', '', notes_part)
        notes_part = notes_part.strip()

        # FIX: dọn luôn ký tự rác còn sót lại ở CUỐI chuỗi
        notes_part = _re.sub(r'[,"\'\s]+$', '', notes_part)
        notes_part = _re.sub(r'^Optimize\s+to\s+', '', notes_part, flags=_re.IGNORECASE)
        notes_part = _re.sub(r'^Optimize\s+', '', notes_part, flags=_re.IGNORECASE)
        notes = notes_part.strip()

    if not notes and title:
        notes = title

    if not title:
        fallback_match = _re.search(
            r'"tasks"\s*[:\[]*\s*"?title"?\s*[:"]+\s*"([^"]+)"',
            raw, flags=_re.IGNORECASE
        )
        if fallback_match:
            title = fallback_match.group(1).strip()

    if not title:
        title = "Task"
    if not notes:
        notes = title

    title = _re.sub(r'\s+', ' ', title.strip())
    notes = _re.sub(r'\s+', ' ', notes.strip())
    notes = notes.rstrip(']}').strip()

    deadline_patterns = [
        r'\s+before\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b',
        r'\s+by\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b',
        r'\s+before\s+(?:today|tomorrow|tonight)\b',
        r'\s+by\s+(?:today|tomorrow|tonight)\b',
        r'\s+before\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\s+by\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    ]
    for pattern in deadline_patterns:
        notes = _re.sub(pattern, '', notes, flags=_re.IGNORECASE)
    notes = notes.strip()

    return {"tasks": [{"title": title, "notes": notes}]}


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictRequest):

    text = request.text.strip()

    if not text:
        return {
            "tasks": [
                {"title": "", "notes": ""}
            ]
        }

    try:

        raw = generate_text(text)

        result = parse_model_output(raw)

        print("\n" + "-" * 70)
        print("INPUT:", text)
        print("RAW MODEL OUTPUT:", raw)
        print("FINAL JSON:", json.dumps(result, ensure_ascii=False))
        print("-" * 70)

        return result

    except Exception as e:

        return {
            "tasks": [
                {"title": "Task", "notes": str(e)}
            ]
        }


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_dir": MODEL_DIR,
        "encoder": os.path.basename(ENCODER_PATH),
        "decoder": os.path.basename(DECODER_PATH),
        "providers": providers,
        "encoder_inputs": get_input_names(encoder_session),
        "encoder_outputs": get_output_names(encoder_session),
        "decoder_inputs": get_input_names(decoder_session),
        "decoder_outputs": get_output_names(decoder_session)
    }