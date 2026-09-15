
import os
import json
import re
import time
import logging

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel


# ============================================================
# LOGGING CONFIG
# ============================================================
# BẢO MẬT:
# Mặc định KHÔNG log nội dung tin nhắn thật.
#
# Chỉ bật DEBUG_LOG=true trên Render nếu cần debug.
# Khi bật, log chỉ hiện preview rút gọn.
# ============================================================

DEBUG_LOG = (
    os.getenv("DEBUG_LOG", "false").lower() == "true"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("task-extractor-onnx")


def preview(text, max_len=40):
    """
    Rút gọn text để log an toàn.
    """
    if not text:
        return ""

    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_len:
        return text

    return text[:max_len] + "…(" + str(len(text)) + " chars)"


# ============================================================
# API KEY AUTH
# ============================================================

API_KEY = os.getenv("TASK_EXTRACTOR_API_KEY", "")


def verify_api_key(x_api_key: str = Header(default="")):
    if not API_KEY:
        # Chưa cấu hình key → giữ hành vi cũ.
        return

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model_onnx_int8"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "encoder_model.onnx"
)

DECODER_PATH = os.path.join(
    MODEL_DIR,
    "decoder_model.onnx"
)

TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer.json"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.json"
)


# ============================================================
# ONNX RUNTIME CONFIG
# ============================================================

# Render Free = CPU/RAM thấp.
SESSION_OPTIONS = ort.SessionOptions()

SESSION_OPTIONS.intra_op_num_threads = 1
SESSION_OPTIONS.inter_op_num_threads = 1

SESSION_OPTIONS.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
)

SESSION_OPTIONS.enable_cpu_mem_arena = False
SESSION_OPTIONS.enable_mem_pattern = False

PROVIDERS = [
    "CPUExecutionProvider"
]


# ============================================================
# GENERATION CONFIG
# ============================================================

# Target lúc training tối đa 128 token.
# Output thực tế của task extractor khá ngắn.
MAX_NEW_TOKENS = 64

REPETITION_PENALTY = 1.0

NO_REPEAT_NGRAM_SIZE = 0


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

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy tokenizer.json: {TOKENIZER_PATH}"
    )

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"Không tìm thấy config.json: {CONFIG_PATH}"
    )


# ============================================================
# TOKENIZER
# ============================================================

# Dùng package tokenizers trực tiếp để giảm RAM.
tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

# QUAN TRỌNG:
# Training V4 dùng MAX_INPUT_LENGTH = 384.
# Production phải khớp với training.
tokenizer.enable_truncation(
    max_length=384
)


with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as f:
    _model_config = json.load(f)


# ============================================================
# ONNX SESSIONS
# ============================================================

logger.info("=" * 70)
logger.info("LOADING ONNX MODEL")
logger.info("=" * 70)

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

logger.info("ONNX models loaded.")


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

DECODER_START_TOKEN_ID = _model_config.get(
    "pad_token_id"
)

if DECODER_START_TOKEN_ID is None:
    DECODER_START_TOKEN_ID = _model_config.get(
        "eos_token_id"
    )

if DECODER_START_TOKEN_ID is None:
    DECODER_START_TOKEN_ID = 0


EOS_TOKEN_ID = _model_config.get(
    "eos_token_id"
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Task Extractor ONNX API",
    version="2.0.0"
)


# ============================================================
# REQUEST
# ============================================================

class PredictRequest(BaseModel):
    text: str


# ============================================================
# STARTUP INFO
# ============================================================

logger.info("=" * 70)
logger.info("TASK EXTRACTOR ONNX V4")
logger.info("=" * 70)

logger.info(
    "Model directory: %s",
    MODEL_DIR
)

logger.info(
    "Encoder inputs: %s",
    ENCODER_INPUT_NAMES
)

logger.info(
    "Encoder outputs: %s",
    ENCODER_OUTPUT_NAMES
)

logger.info(
    "Decoder inputs: %s",
    DECODER_INPUT_NAMES
)

logger.info(
    "Decoder outputs: %s",
    DECODER_OUTPUT_NAMES
)

logger.info(
    "MAX_NEW_TOKENS: %s",
    MAX_NEW_TOKENS
)

logger.info(
    "MAX_INPUT_LENGTH: %s",
    384
)

logger.info(
    "intra_op_num_threads: %s",
    SESSION_OPTIONS.intra_op_num_threads
)

logger.info(
    "inter_op_num_threads: %s",
    SESSION_OPTIONS.inter_op_num_threads
)

logger.info(
    "pad_token_id (decoder start): %s",
    DECODER_START_TOKEN_ID
)

logger.info(
    "eos_token_id: %s",
    EOS_TOKEN_ID
)

logger.info(
    "DEBUG_LOG=%s | API key protection=%s",
    DEBUG_LOG,
    "ENABLED"
    if API_KEY
    else "DISABLED (no TASK_EXTRACTOR_API_KEY set)"
)

logger.info("=" * 70)


# ============================================================
# PROMPT
# ============================================================
#
# QUAN TRỌNG:
# Prompt này phải khớp với build_prompt() trong training V4.
#
# Training:
#   input  = prompt dưới đây
#   target = flat text:
#
#   TITLE: ...
#   NOTES: ...
#   DEADLINE: ...
#
# Production phải dùng đúng format này.
# ============================================================

def build_prompt(text):
    return (
        "Convert the following English chat message into exactly ONE task.\n\n"

        "Rules:\n"

        "1. Create one short, clear, actionable title.\n"

        "2. Notes must preserve the important information from the chat.\n"

        "3. Do not invent information.\n"

        "4. Preserve all names, identifiers, codes, and numbers exactly as "
        "written, in whatever position they appear in the sentence. Never "
        "drop or reposition them.\n"

        "5. Do not include deadlines or due times in the title or notes.\n"

        "6. If the chat contains a deadline or due time, extract it into "
        "the deadline field and preserve its wording exactly as written.\n"

        "7. If the chat does NOT contain a deadline or due time, DO NOT "
        "create a deadline field.\n"

        "8. Do not create multiple tasks. Combine multiple actions into "
        "one task.\n"

        "9. You may fix spelling and grammar mistakes only. Do NOT change "
        "the action verb, do NOT paraphrase, and do NOT replace any word "
        "with a synonym.\n"

        "10. Return output in exactly this plain text format:\n"

        "TITLE: <title>\n"
        "NOTES: <notes>\n"
        "DEADLINE: <deadline>\n"

        "or, when there is no deadline, omit the DEADLINE line entirely.\n\n"

        "Chat:\n"
        f"{text}\n\n"

        "Output:\n"
    )


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

    encoded = tokenizer.encode(prompt)

    input_ids = np.array(
        [encoded.ids],
        dtype=np.int64
    )

    attention_mask = np.array(
        [encoded.attention_mask],
        dtype=np.int64
    )

    tokenize_time = (
        time.perf_counter()
        - tokenize_start
    )


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

    encoder_time = (
        time.perf_counter()
        - encoder_start
    )


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


        # Current ONNX decoder:
        #
        # encoder_attention_mask
        # input_ids
        # encoder_hidden_states
        #
        # Keep support for alternative exported names.

        for name in DECODER_INPUT_NAMES:

            if name in (
                "decoder_input_ids",
                "decoder_input_ids_0"
            ):
                decoder_inputs[name] = (
                    decoder_input_ids
                )

            elif name in (
                "encoder_hidden_states",
                "encoder_hidden_states_0"
            ):
                decoder_inputs[name] = (
                    encoder_hidden_states
                )

            elif name == "encoder_attention_mask":
                decoder_inputs[name] = (
                    attention_mask
                )

            elif name == "input_ids":
                decoder_inputs[name] = (
                    decoder_input_ids
                )

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
                        score
                        / repetition_penalty
                    )

                else:

                    next_token_logits[
                        0,
                        token_id
                    ] = (
                        score
                        * repetition_penalty
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
        # GREEDY DECODING
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


    decoder_time = (
        time.perf_counter()
        - decoder_start
    )


    # --------------------------------------------------------
    # FINAL DECODE
    # --------------------------------------------------------

    result = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    total_time = (
        time.perf_counter()
        - total_start
    )


    # --------------------------------------------------------
    # TIMING LOG
    # --------------------------------------------------------

    logger.info(
        "[TIMING] tokenize=%.3fs | encoder=%.3fs | "
        "decoder=%.3fs | total=%.3fs | tokens=%d",
        tokenize_time,
        encoder_time,
        decoder_time,
        total_time,
        len(generated)
    )


    return result.strip()


# ============================================================
# FLAT TEXT PARSER
# ============================================================
#
# Model output:
#
# TITLE: Review the pull request
# NOTES: Review the pull request.
# DEADLINE: next Friday
#
# API response:
#
# {
#   "tasks": [
#     {
#       "title": "...",
#       "notes": "...",
#       "deadline": "..."
#     }
#   ]
# }
#
# Deadline được giữ nguyên raw text.
# KHÔNG resolve / normalize / calculate ở đây.
# ============================================================

def parse_model_output(raw_output: str):

    if raw_output is None:
        raw_output = ""

    raw = str(raw_output).strip()


    # --------------------------------------------------------
    # REMOVE SPECIAL TOKENS
    # --------------------------------------------------------

    raw = raw.replace(
        "<pad>",
        ""
    )

    raw = raw.replace(
        "</s>",
        ""
    )

    raw = raw.replace(
        "<s>",
        ""
    )

    raw = raw.strip()


    # --------------------------------------------------------
    # NORMALIZE WHITESPACE
    # --------------------------------------------------------

    raw = re.sub(
        r"\s+",
        " ",
        raw
    ).strip()


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------
    #
    # Everything between TITLE: and NOTES:/DEADLINE:/end.
    # --------------------------------------------------------

    title_match = re.search(
        r"\bTITLE\s*:\s*(.*?)(?=\s+NOTES\s*:|\s+DEADLINE\s*:|$)",
        raw,
        flags=re.IGNORECASE
    )

    title = ""

    if title_match:
        title = title_match.group(1).strip()


    # --------------------------------------------------------
    # NOTES
    # --------------------------------------------------------
    #
    # Everything between NOTES: and DEADLINE:/end.
    # --------------------------------------------------------

    notes_match = re.search(
        r"\bNOTES\s*:\s*(.*?)(?=\s+DEADLINE\s*:|$)",
        raw,
        flags=re.IGNORECASE
    )

    notes = ""

    if notes_match:
        notes = notes_match.group(1).strip()


    # --------------------------------------------------------
    # DEADLINE
    # --------------------------------------------------------
    #
    # Preserve EXACTLY what the model generated after
    # DEADLINE:.
    #
    # Do not resolve:
    #   next Friday
    #   tomorrow morning
    #   tonight
    #   EOQ
    #   etc.
    #
    # DeadlineParser in Apps Script handles resolution later.
    # --------------------------------------------------------

    deadline_match = re.search(
        r"\bDEADLINE\s*:\s*(.*)$",
        raw,
        flags=re.IGNORECASE
    )

    deadline = ""

    if deadline_match:
        deadline = deadline_match.group(1).strip()


    # --------------------------------------------------------
    # CLEAN TRAILING PUNCTUATION ONLY
    # --------------------------------------------------------
    #
    # Do NOT alter wording.
    #
    # In particular, do not:
    #   - paraphrase
    #   - replace synonyms
    #   - invent information
    #   - modify dates
    # --------------------------------------------------------

    title = re.sub(
        r"\s+",
        " ",
        title
    ).strip()

    notes = re.sub(
        r"\s+",
        " ",
        notes
    ).strip()

    deadline = re.sub(
        r"\s+",
        " ",
        deadline
    ).strip()


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------
    #
    # Do not invent a deadline.
    #
    # If model output is malformed:
    # - preserve whatever valid title/notes we got
    # - deadline remains empty
    #
    # This is safer than copying input into deadline.
    # --------------------------------------------------------

    if not title:
        title = "Task"

    if not notes:
        notes = title


    return {
        "tasks": [
            {
                "title": title,
                "notes": notes,
                "deadline": deadline
            }
        ]
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    request: PredictRequest,
    _auth=Depends(verify_api_key)
):

    text = request.text.strip()


    # --------------------------------------------------------
    # EMPTY INPUT
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # GENERATE
        # ----------------------------------------------------

        raw = generate_text(
            text
        )


        # ----------------------------------------------------
        # PARSE FLAT OUTPUT
        # ----------------------------------------------------

        result = parse_model_output(
            raw
        )


        # ----------------------------------------------------
        # DEBUG LOG
        # ----------------------------------------------------

        if DEBUG_LOG:

            logger.info(
                "-" * 70
            )

            logger.info(
                "INPUT: %s",
                preview(text)
            )

            logger.info(
                "RAW MODEL OUTPUT: %s",
                preview(
                    raw,
                    max_len=120
                )
            )

            logger.info(
                "PARSED TITLE: %s",
                preview(
                    result["tasks"][0].get(
                        "title",
                        ""
                    ),
                    max_len=60
                )
            )

            logger.info(
                "PARSED NOTES: %s",
                preview(
                    result["tasks"][0].get(
                        "notes",
                        ""
                    ),
                    max_len=80
                )

            )

            logger.info(
                "PARSED DEADLINE: %s",
                preview(
                    result["tasks"][0].get(
                        "deadline",
                        ""
                    ),
                    max_len=60
                )
            )

            logger.info(
                "-" * 70
            )

        else:

            logger.info(
                "Predict request handled. "
                "input_len=%d tasks=%d",
                len(text),
                len(
                    result.get(
                        "tasks",
                        []
                    )
                )
            )


        return result


    except Exception as e:

        # ----------------------------------------------------
        # INTERNAL ERROR
        # ----------------------------------------------------

        # Không trả nội dung input/raw model output về client.

        logger.error(
            "[ERROR] %s",
            repr(e)
        )

        return {
            "tasks": [
                {
                    "title": "Task",
                    "notes": (
                        "Internal error while "
                        "processing request."
                    )
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
        "model": (
            "task_extractor_v3_"
            "flan_t5_small_v4"
        ),
        "backend": "ONNX Runtime"
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info(
    _auth=Depends(verify_api_key)
):

    return {

        "model_dir": MODEL_DIR,

        "encoder": os.path.basename(
            ENCODER_PATH
        ),

        "decoder": os.path.basename(
            DECODER_PATH
        ),

        "providers": PROVIDERS,

        "encoder_inputs": (
            ENCODER_INPUT_NAMES
        ),

        "encoder_outputs": (
            ENCODER_OUTPUT_NAMES
        ),

        "decoder_inputs": (
            DECODER_INPUT_NAMES
        ),

        "decoder_outputs": (
            DECODER_OUTPUT_NAMES
        ),

        "max_input_length": 384,

        "max_new_tokens": (
            MAX_NEW_TOKENS
        ),

        "intra_op_num_threads": (
            SESSION_OPTIONS
            .intra_op_num_threads
        ),

        "inter_op_num_threads": (
            SESSION_OPTIONS
            .inter_op_num_threads
        )
    }
