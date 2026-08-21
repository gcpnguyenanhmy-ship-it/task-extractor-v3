import os
import re
import json
import zipfile
import shutil
import logging

import torch

from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# LOGGING CONFIG
#
# BẢO MẬT:
# Mặc định KHÔNG log nội dung tin nhắn thật (INPUT / RAW MODEL
# OUTPUT / FINAL JSON) ra Render logs, vì đây là dữ liệu nội bộ
# nhạy cảm (công việc, tên người, deadline...).
#
# Chỉ bật xem chi tiết bằng cách set biến môi trường trên Render:
#   DEBUG_LOG = true
#
# Khi bật, log cũng chỉ hiện preview rút gọn (không toàn văn).
# ============================================================

DEBUG_LOG = (
    os.getenv("DEBUG_LOG", "false").lower() == "true"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("task-extractor")


def preview(text, max_len=40):
    """
    Rút gọn text để log an toàn — không lộ toàn bộ nội dung.
    """

    if not text:
        return ""

    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_len:
        return text

    return text[:max_len] + "…(" + str(len(text)) + " chars)"


# ============================================================
# API KEY AUTH
#
# BẢO MẬT:
# /predict trước đây không yêu cầu xác thực — bất kỳ ai biết URL
# cũng gọi được. Thêm API key dùng chung (giống RESULT_TOKEN bên
# Cloudflare Worker) để chỉ hệ thống của bạn gọi được.
#
# Set biến môi trường trên Render:
#   TASK_EXTRACTOR_API_KEY = <chuỗi bí mật dài, ngẫu nhiên>
#
# Nếu không set biến này, endpoint sẽ mở public như cũ (để không
# làm gãy hệ thống nếu bạn chưa kịp cấu hình) — NÊN set trong
# production.
# ============================================================

API_KEY = os.getenv("TASK_EXTRACTOR_API_KEY", "")


def verify_api_key(x_api_key: str = Header(default="")):

    if not API_KEY:
        # Chưa cấu hình key → không chặn (giữ hành vi cũ).
        return

    if x_api_key != API_KEY:

        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ZIP_FILE = os.path.join(
    BASE_DIR,
    "task_extractor_v3_flan_t5_small.zip"
)

MODEL_EXTRACT_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MAX_INPUT_LENGTH = 256
MAX_NEW_TOKENS = 96

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

logger.info("=" * 70)
logger.info("TASK EXTRACTOR V3 API")
logger.info("=" * 70)

logger.info("Base directory: %s", BASE_DIR)
logger.info("Device: %s", DEVICE)

if torch.cuda.is_available():

    logger.info(
        "GPU: %s",
        torch.cuda.get_device_name(0)
    )

logger.info(
    "DEBUG_LOG=%s | API key protection=%s",
    DEBUG_LOG,
    "ENABLED" if API_KEY else "DISABLED (no TASK_EXTRACTOR_API_KEY set)"
)


# ============================================================
# CHECK ZIP
# ============================================================

if not os.path.exists(ZIP_FILE):

    raise FileNotFoundError(
        f"\nKhông tìm thấy model ZIP:\n{ZIP_FILE}\n\n"
        "Hãy đặt task_extractor_v3_flan_t5_small.zip "
        "cùng thư mục với app.py."
    )


# ============================================================
# EXTRACT MODEL
# ============================================================

logger.info("=" * 70)
logger.info("CHECK MODEL")
logger.info("=" * 70)


def find_config(directory):

    for root, dirs, files in os.walk(
        directory
    ):

        if "config.json" in files:

            return os.path.join(
                root,
                "config.json"
            )

    return None


config_file = find_config(
    MODEL_EXTRACT_DIR
)


# ------------------------------------------------------------
# Extract nếu chưa có model
# ------------------------------------------------------------

if config_file is None:

    logger.info("Extracting model ZIP...")

    if os.path.exists(
        MODEL_EXTRACT_DIR
    ):

        shutil.rmtree(
            MODEL_EXTRACT_DIR
        )

    os.makedirs(
        MODEL_EXTRACT_DIR,
        exist_ok=True
    )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as z:

        z.extractall(
            MODEL_EXTRACT_DIR
        )

    logger.info("Extract complete.")

else:

    logger.info("Model already extracted.")


# ============================================================
# FIND REAL MODEL DIRECTORY
# ============================================================

logger.info("=" * 70)
logger.info("FIND MODEL DIRECTORY")
logger.info("=" * 70)


def find_model_directory(root):

    for current_root, dirs, files in os.walk(
        root
    ):

        if "config.json" not in files:
            continue

        has_weights = False

        for file in files:

            if (
                file.endswith(".safetensors")
                or file.endswith(".bin")
            ):

                has_weights = True
                break

        if has_weights:

            return current_root

    return None


MODEL_DIR = find_model_directory(
    MODEL_EXTRACT_DIR
)


if MODEL_DIR is None:

    logger.error("Files found in ZIP:")

    for root, dirs, files in os.walk(
        MODEL_EXTRACT_DIR
    ):

        for file in files:

            logger.error(
                os.path.join(
                    root,
                    file
                )
            )

    raise RuntimeError(
        "\nKhông tìm thấy config.json + "
        "model weights trong ZIP."
    )


logger.info("Model directory: %s", MODEL_DIR)


# ============================================================
# LOAD TOKENIZER
# ============================================================

logger.info("=" * 70)
logger.info("LOADING TOKENIZER")
logger.info("=" * 70)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


logger.info("Tokenizer loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

logger.info("=" * 70)
logger.info("LOADING MODEL")
logger.info("=" * 70)


model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)


model.to(
    DEVICE
)

model.eval()


logger.info("Model loaded successfully.")
logger.info("Device: %s", DEVICE)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Task Extractor V3 API",
    version="1.0.0"
)


# ============================================================
# REQUEST
# ============================================================

class TaskRequest(BaseModel):

    text: str


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
# JSON PARSER
# ============================================================

def parse_model_output(raw_output: str):
    """
    Robust parser for FLAN-T5 output.

    Expected final format:

    {
        "tasks": [
            {
                "title": "...",
                "notes": "..."
            }
        ]
    }

    The model may generate malformed JSON such as:

    "tasks":["title":"...","notes"Optimize to ...

    This parser tries to recover title + notes instead of
    returning HTTP 500.
    """

    import json
    import re

    # =========================================================
    # CLEAN RAW OUTPUT
    # =========================================================

    if raw_output is None:
        raw_output = ""

    raw = str(raw_output).strip()

    # Remove common special tokens
    raw = raw.replace("<pad>", "")
    raw = raw.replace("</s>", "")
    raw = raw.replace("<s>", "")
    raw = raw.strip()

    # Normalize smart quotes
    raw = (
        raw.replace("“", '"')
           .replace("”", '"')
           .replace("‘", "'")
           .replace("’", "'")
    )

    # =========================================================
    # 1. TRY NORMAL JSON FIRST
    # =========================================================

    try:
        data = json.loads(raw)

        # ---------------------------------------------
        # Expected:
        # {"tasks":[{"title":"...","notes":"..."}]}
        # ---------------------------------------------

        if isinstance(data, dict):

            tasks = data.get("tasks")

            if isinstance(tasks, list) and len(tasks) > 0:

                first_task = tasks[0]

                if isinstance(first_task, dict):

                    title = str(
                        first_task.get("title", "")
                    ).strip()

                    notes = str(
                        first_task.get("notes", "")
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

    # =========================================================
    # 2. EXTRACT TITLE
    # =========================================================

    title = ""

    title_patterns = [

        # "title":"..."
        r'"title"\s*:\s*"([^"]+)"',

        # "title":"..."
        # with optional malformed syntax
        r'"title"\s*"[^"]*"\s*:\s*"([^"]+)"',

        # title: "..."
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

    # =========================================================
    # 3. EXTRACT NOTES
    # =========================================================

    notes = ""

    # ---------------------------------------------------------
    # Find where "notes" starts
    # ---------------------------------------------------------

    notes_match = re.search(
        r'"notes"',
        raw,
        flags=re.IGNORECASE
    )

    if notes_match:

        notes_start = notes_match.end()

        notes_part = raw[notes_start:]

        # -----------------------------------------------------
        # Remove malformed tokens between "notes" and content
        #
        # Examples:
        #
        # "notes"Optimize":
        # "notes"Optimize to
        # "notes":
        # -----------------------------------------------------

        notes_part = re.sub(
            r'^\s*'
            r'(?:'
            r'[:\-]?\s*'
            r'(?:Optimize|optimized|optimization)'
            r'(?:\s+to)?'
            r')?'
            r'\s*[:\-]?\s*',
            '',
            notes_part,
            flags=re.IGNORECASE
        )

        # -----------------------------------------------------
        # Remove leading quote
        # -----------------------------------------------------

        notes_part = notes_part.lstrip()

        if notes_part.startswith('"'):
            notes_part = notes_part[1:]

        notes_part = notes_part.strip()

        # -----------------------------------------------------
        # Remove ending JSON characters
        # -----------------------------------------------------

        notes_part = re.sub(
            r'"\s*\]?\s*\}?\s*$',
            '',
            notes_part
        )

        notes_part = notes_part.strip()

        # -----------------------------------------------------
        # Sometimes model outputs:
        #
        # notes"Optimize to export...
        #
        # We don't want "Optimize to".
        # -----------------------------------------------------

        notes_part = re.sub(
            r'^Optimize\s+to\s+',
            '',
            notes_part,
            flags=re.IGNORECASE
        )

        notes_part = re.sub(
            r'^Optimize\s+',
            '',
            notes_part,
            flags=re.IGNORECASE
        )

        notes = notes_part.strip()

    # =========================================================
    # 4. FALLBACK NOTES
    # =========================================================

    # If notes extraction failed but title exists,
    # use title as notes.
    #
    # This prevents API 500.
    #
    # Example:
    #
    # title = "Check D06"
    # notes = ""
    #
    # becomes:
    #
    # notes = "Check D06"
    # =========================================================

    if not notes and title:

        notes = title

    # =========================================================
    # 5. FALLBACK TITLE
    # =========================================================

    # If title cannot be extracted, try to find
    # something after "tasks".
    # =========================================================

    if not title:

        fallback_match = re.search(
            r'"tasks"\s*[:\[]*\s*'
            r'"?title"?\s*[:"]+\s*'
            r'"([^"]+)"',
            raw,
            flags=re.IGNORECASE
        )

        if fallback_match:

            title = fallback_match.group(1).strip()

    # =========================================================
    # 6. FINAL FALLBACK
    # =========================================================

    if not title:

        # Never crash the API.
        title = "Task"

    if not notes:

        notes = title

    # =========================================================
    # 7. CLEAN TITLE
    # =========================================================

    title = title.strip()

    title = re.sub(
        r'\s+',
        ' ',
        title
    )

    # =========================================================
    # 8. CLEAN NOTES
    # =========================================================

    notes = notes.strip()

    notes = re.sub(
        r'\s+',
        ' ',
        notes
    )

    # Remove accidental trailing JSON characters
    notes = notes.rstrip(']}')

    notes = notes.strip()

    # =========================================================
    # 9. REMOVE DEADLINE FROM NOTES
    # =========================================================

    # IMPORTANT:
    # This is only a safety cleanup.
    #
    # The model should already have learned
    # not to include deadlines.
    # =========================================================

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

    # =========================================================
    # 10. FINAL RESULT
    # =========================================================

    result = {
        "tasks": [
            {
                "title": title,
                "notes": notes
            }
        ]
    }

    return result


# ============================================================
# MODEL INFERENCE
# ============================================================

@torch.inference_mode()
def predict_task(text):

    prompt = build_prompt(
        text
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )


    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }


    output_ids = model.generate(

        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=False,

        num_beams=1,

        repetition_penalty=1.05,

        no_repeat_ngram_size=3
    )


    raw = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )


    raw = raw.strip()


    result = parse_model_output(
        raw
    )


    return raw, result


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "model": "task-extractor-v3-flan-t5-small",
        "device": DEVICE
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# PREDICT
#
# BẢO MẬT:
# - Yêu cầu header "x-api-key" khớp TASK_EXTRACTOR_API_KEY
#   (nếu biến này đã được set trên Render).
# - KHÔNG log toàn văn text/raw output ra console theo mặc định.
#   Chỉ log khi DEBUG_LOG=true, và chỉ log bản preview rút gọn.
# ============================================================

@app.post("/predict")
def predict(
    request: TaskRequest,
    _auth=Depends(verify_api_key)
):

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


    raw, result = predict_task(
        text
    )


    if DEBUG_LOG:

        logger.info("-" * 70)
        logger.info("INPUT: %s", preview(text))
        logger.info("RAW MODEL OUTPUT: %s", preview(raw, max_len=80))
        logger.info(
            "FINAL JSON keys: tasks=%d",
            len(result.get("tasks", []))
        )
        logger.info("-" * 70)

    else:

        logger.info(
            "Predict request handled. input_len=%d tasks=%d",
            len(text),
            len(result.get("tasks", []))
        )


    return result