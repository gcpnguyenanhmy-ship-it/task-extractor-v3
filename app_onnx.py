import os
import json
import re
import time

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

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "encoder_model.onnx"
)

DECODER_PATH = os.path.join(
    MODEL_DIR,
    "decoder_model.onnx"
)

# Render Free = CPU rất thấp
# Không nên để ONNX tự tạo quá nhiều CPU threads.
SESSION_OPTIONS = ort.SessionOptions()

SESSION_OPTIONS.intra_op_num_threads = 1
SESSION_OPTIONS.inter_op_num_threads = 1

SESSION_OPTIONS.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

PROVIDERS = ["CPUExecutionProvider"]

# Giới hạn output để giảm số lần decoder.run()
MAX_NEW_TOKENS = 48

REPETITION_PENALTY = 1.05
NO_REPEAT_NGRAM_SIZE = 3


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

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ============================================================
# ONNX SESSIONS
# ============================================================

print("=" * 70)
print("LOADING ONNX MODEL")
print("=" * 70)

encoder_session = ort.InferenceSession(
    ENCODER_PATH,
    sess_options=SESSION_OPTIONS,
    providers=PROVIDERS
)

decoder_session = ort.InferenceSession(
    DECODER_PATH,
    sess_options=SESSION_OPTIONS,
    providers=PROVIDERS
)

print("ONNX models loaded.")


# ============================================================
# CACHE MODEL METADATA
# ============================================================

ENCODER_INPUT_NAMES = [
    x.name
    for x in encoder_session.get_inputs()
]

ENCODER_OUTPUT_NAMES = [
    x.name
    for x in encoder_session.get_outputs()
]

DECODER_INPUT_NAMES = [
    x.name
    for x in decoder_session.get_inputs()
]

DECODER_OUTPUT_NAMES = [
    x.name
    for x in decoder_session.get_outputs()
]


# ============================================================
# TOKEN IDS
# ============================================================

DECODER_START_TOKEN_ID = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)

if DECODER_START_TOKEN_ID is None:
    DECODER_START_TOKEN_ID = 0


EOS_TOKEN_ID = tokenizer.eos_token_id


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Task Extractor ONNX API",
    version="1.1.0"
)


# ============================================================
# REQUEST
# ============================================================

class PredictRequest(BaseModel):
    text: str


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

print("\nGeneration:")
print(" MAX_NEW_TOKENS:", MAX_NEW_TOKENS)
print(" intra_op_num_threads:", SESSION_OPTIONS.intra_op_num_threads)
print(" inter_op_num_threads:", SESSION_OPTIONS.inter_op_num_threads)

print("=" * 70)


# ============================================================
# PROMPT
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
# JSON COMPLETION CHECK
# ============================================================

def looks_like_complete_json(text):

    text = text.strip()

    if not text:
        return False

    # Phải có cấu trúc cơ bản
    if not text.startswith("{"):
        return False

    if '"tasks"' not in text:
        return False

    if '"title"' not in text:
        return False

    if '"notes"' not in text:
        return False

    # JSON hoàn chỉnh
    try:
        data = json.loads(text)

        if (
            isinstance(data, dict)
            and isinstance(data.get("tasks"), list)
            and len(data["tasks"]) > 0
        ):
            task = data["tasks"][0]

            if (
                isinstance(task, dict)
                and str(task.get("title", "")).strip()
                and str(task.get("notes", "")).strip()
            ):
                return True

    except Exception:
        pass

    return False


# ============================================================
# GENERATION
# ============================================================

def generate_text(
    text,
    max_new_tokens=MAX_NEW_TOKENS,
    repetition_penalty=REPETITION_PENALTY,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE
):

    total_start = time.perf_counter()

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    prompt = build_prompt(text)

    tokenize_start = time.perf_counter()

    encoded = tokenizer(
        prompt,
        return_tensors="np",
        padding=False,
        truncation=True,
        max_length=512
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    tokenize_time = time.perf_counter() - tokenize_start

    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    encoder_start = time.perf_counter()

    encoder_inputs = {}

    if "input_ids" in ENCODER_INPUT_NAMES:
        encoder_inputs["input_ids"] = input_ids

    if "attention_mask" in ENCODER_INPUT_NAMES:
        encoder_inputs["attention_mask"] = attention_mask

    encoder_outputs = encoder_session.run(
        None,
        encoder_inputs
    )

    encoder_hidden_states = encoder_outputs[0]

    encoder_time = time.perf_counter() - encoder_start

    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    decoder_start = time.perf_counter()

    generated = [
        DECODER_START_TOKEN_ID
    ]

    for _ in range(max_new_tokens):

        decoder_input_ids = np.array(
            [generated],
            dtype=np.int64
        )

        decoder_inputs = {}

        for name in DECODER_INPUT_NAMES:

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

            elif name == "input_ids":
                decoder_inputs[name] = decoder_input_ids

            elif name == "attention_mask":
                decoder_inputs[name] = np.ones_like(
                    decoder_input_ids
                )

        outputs = decoder_session.run(
            None,
            decoder_inputs
        )

        logits = outputs[0]

        next_token_logits = (
            logits[:, -1, :].copy()
        )

        # ----------------------------------------------------
        # REPETITION PENALTY
        # ----------------------------------------------------

        if repetition_penalty != 1.0:

            for token_id in set(generated):

                score = next_token_logits[
                    0,
                    token_id
                ]

                if score > 0:

                    next_token_logits[
                        0,
                        token_id
                    ] = (
                        score /
                        repetition_penalty
                    )

                else:

                    next_token_logits[
                        0,
                        token_id
                    ] = (
                        score *
                        repetition_penalty
                    )

        # ----------------------------------------------------
        # NO REPEAT NGRAM
        # ----------------------------------------------------

        if (
            no_repeat_ngram_size
            and
            len(generated) >= no_repeat_ngram_size
        ):

            n = no_repeat_ngram_size

            prev_ngrams = {}

            for i in range(
                len(generated) - n + 1
            ):

                ngram = tuple(
                    generated[
                        i:i + n - 1
                    ]
                )

                next_tok = generated[
                    i + n - 1
                ]

                prev_ngrams.setdefault(
                    ngram,
                    set()
                ).add(next_tok)

            current_prefix = tuple(
                generated[-(n - 1):]
            )

            banned_tokens = prev_ngrams.get(
                current_prefix,
                set()
            )

            for token_id in banned_tokens:

                next_token_logits[
                    0,
                    token_id
                ] = -1e9

        # ----------------------------------------------------
        # GREEDY
        # ----------------------------------------------------

        next_token_id = int(
            np.argmax(
                next_token_logits,
                axis=-1
            )[0]
        )

        generated.append(
            next_token_id
        )

        # ----------------------------------------------------
        # EOS
        # ----------------------------------------------------

        if (
            EOS_TOKEN_ID is not None
            and
            next_token_id == EOS_TOKEN_ID
        ):
            break

        # ----------------------------------------------------
        # EARLY STOP
        # ----------------------------------------------------

        # Decode phần đang có để kiểm tra JSON.
        # Chỉ kiểm tra khi output đã đủ dài.
        if len(generated) >= 12:

            partial = tokenizer.decode(
                generated,
                skip_special_tokens=True
            )

            if looks_like_complete_json(partial):
                break

    decoder_time = time.perf_counter() - decoder_start

    # --------------------------------------------------------
    # DECODE FINAL
    # --------------------------------------------------------

    output_ids = np.array(
        [generated],
        dtype=np.int64
    )

    result = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    total_time = time.perf_counter() - total_start

    print(
        "[TIMING] "
        f"tokenize={tokenize_time:.3f}s | "
        f"encoder={encoder_time:.3f}s | "
        f"decoder={decoder_time:.3f}s | "
        f"total={total_time:.3f}s | "
        f"tokens={len(generated)}"
    )

    return result.strip()


# ============================================================
# JSON PARSER
# ============================================================

def parse_model_output(raw_output: str):

    if raw_output is None:
        raw_output = ""

    raw = str(raw_output).strip()

    raw = raw.replace("<pad>", "")
    raw = raw.replace("</s>", "")
    raw = raw.replace("<s>", "")

    raw = raw.strip()

    raw = (
        raw
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    # --------------------------------------------------------
    # DIRECT JSON
    # --------------------------------------------------------

    try:

        data = json.loads(raw)

        if isinstance(data, dict):

            tasks = data.get("tasks")

            if (
                isinstance(tasks, list)
                and len(tasks) > 0
            ):

                first_task = tasks[0]

                if isinstance(first_task, dict):

                    title = str(
                        first_task.get(
                            "title",
                            ""
                        )
                    ).strip()

                    notes = str(
                        first_task.get(
                            "notes",
                            ""
                        )
                    ).strip()

                    if title and notes:

                        return {
                            "tasks": [
                                {
                                    "title": title,
                                    "notes": notes
                                }
                            ]
                        }

    except Exception:
        pass

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    title = ""

    title_patterns = [
        r'"title"\s*:\s*"([^"]+)"',
        r'title\s*:\s*"([^"]+)"',
    ]

    for pattern in title_patterns:

        match = re.search(
            pattern,
            raw,
            flags=re.IGNORECASE
        )

        if match:

            title = match.group(1).strip()

            if title:
                break

    notes = ""

    notes_match = re.search(
        r'"notes"',
        raw,
        flags=re.IGNORECASE
    )

    if notes_match:

        notes_part = raw[
            notes_match.end():
        ]

        notes_part = re.sub(
            r'^\s*:\s*',
            '',
            notes_part
        )

        notes_part = notes_part.lstrip(
            ' \t\r\n:,"\''
        )

        notes_part = re.sub(
            r'"\s*\]?\s*\}?\s*$',
            '',
            notes_part
        )

        notes_part = re.sub(
            r'[,"\']+\s*$',
            '',
            notes_part
        )

        notes = notes_part.strip()

    if not notes and title:
        notes = title

    if not title:
        title = "Task"

    if not notes:
        notes = title

    title = re.sub(
        r'\s+',
        ' ',
        title.strip()
    )

    notes = re.sub(
        r'\s+',
        ' ',
        notes.strip()
    )

    notes = notes.rstrip(
        ']}'
    ).strip()

    # --------------------------------------------------------
    # REMOVE DEADLINE FROM NOTES
    # --------------------------------------------------------

    deadline_patterns = [

        r'\s+before\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b',

        r'\s+by\s+\d{1,2}(?::\d{2})?\s*(?:AM|PM)\b',

        r'\s+before\s+(?:today|tomorrow|tonight)\b',

        r'\s+by\s+(?:today|tomorrow|tonight)\b',

        r'\s+before\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',

        r'\s+by\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    ]

    for pattern in deadline_patterns:

        notes = re.sub(
            pattern,
            '',
            notes,
            flags=re.IGNORECASE
        )

    notes = notes.strip()

    return {
        "tasks": [
            {
                "title": title,
                "notes": notes
            }
        ]
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictRequest):

    text = request.text.strip()

    if not text:

        return {
            "tasks": [
                {
                    "title": "",
                    "notes": ""
                }
            ]
        }

    try:

        raw = generate_text(text)

        result = parse_model_output(
            raw
        )

        print("\n" + "-" * 70)
        print("INPUT:", text)
        print(
            "RAW MODEL OUTPUT:",
            raw
        )
        print(
            "FINAL JSON:",
            json.dumps(
                result,
                ensure_ascii=False
            )
        )
        print("-" * 70)

        return result

    except Exception as e:

        print(
            "[ERROR]",
            repr(e)
        )
        return {
            "tasks": [
                {
                    "title": "Task",
                    "notes": str(e)
                }
            ]
        }
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
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_dir": MODEL_DIR,
        "encoder": os.path.basename( ENCODER_PATH ),
        "decoder": os.path.basename( DECODER_PATH),
        "providers": PROVIDERS,
        "encoder_inputs":ENCODER_INPUT_NAMES,
        "encoder_outputs":ENCODER_OUTPUT_NAMES,
        "decoder_inputs":DECODER_INPUT_NAMES,
        "decoder_outputs":DECODER_OUTPUT_NAMES,
        "max_new_tokens":MAX_NEW_TOKENS,
        "intra_op_num_threads":SESSION_OPTIONS.intra_op_num_threads,
        "inter_op_num_threads":SESSION_OPTIONS.inter_op_num_threads
    }