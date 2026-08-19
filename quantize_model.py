"""
Quantize model ONNX từ float32 sang INT8 (dynamic quantization).

Mục đích: giảm RAM + dung lượng file, để vừa giới hạn 512MB
RAM của Render free tier.

Cách dùng:
1. Cài thư viện (nếu chưa có):
   pip install onnxruntime

2. Đặt file này CÙNG cấp với thư mục model_onnx/ (chứa
   encoder_model.onnx + decoder_model.onnx bản gốc float32).

3. Chạy:
   python quantize_model.py

4. Kết quả: thư mục model_onnx_int8/ chứa bản đã quantize.
   Dùng thư mục NÀY để deploy (đổi MODEL_DIR trong
   app_onnx_manual.py sang "model_onnx_int8").
"""

import os
import shutil

from onnxruntime.quantization import quantize_dynamic, QuantType

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.join(BASE_DIR, "model_onnx")
DST_DIR = os.path.join(BASE_DIR, "model_onnx_int8")

if not os.path.exists(SRC_DIR):
    raise FileNotFoundError(
        f"Không tìm thấy thư mục nguồn: {SRC_DIR}"
    )

os.makedirs(DST_DIR, exist_ok=True)

print("=" * 60)
print("QUANTIZE MODEL ONNX: float32 -> INT8 (dynamic)")
print("=" * 60)


def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# ============================================================
# QUANTIZE ENCODER
# ============================================================

encoder_src = os.path.join(SRC_DIR, "encoder_model.onnx")
encoder_dst = os.path.join(DST_DIR, "encoder_model.onnx")

print(f"\nEncoder gốc: {get_size_mb(encoder_src):.1f} MB")

quantize_dynamic(
    model_input=encoder_src,
    model_output=encoder_dst,
    weight_type=QuantType.QUInt8
)

print(f"Encoder sau quantize: {get_size_mb(encoder_dst):.1f} MB")


# ============================================================
# QUANTIZE DECODER
# ============================================================

decoder_src = os.path.join(SRC_DIR, "decoder_model.onnx")
decoder_dst = os.path.join(DST_DIR, "decoder_model.onnx")

print(f"\nDecoder gốc: {get_size_mb(decoder_src):.1f} MB")

quantize_dynamic(
    model_input=decoder_src,
    model_output=decoder_dst,
    weight_type=QuantType.QUInt8
)

print(f"Decoder sau quantize: {get_size_mb(decoder_dst):.1f} MB")


# ============================================================
# COPY CÁC FILE KHÔNG CẦN QUANTIZE
# (tokenizer, config, generation_config, spiece.model...)
# ============================================================

print("\nCopy các file phụ trợ (tokenizer, config...)")

for filename in os.listdir(SRC_DIR):

    if filename in ("encoder_model.onnx", "decoder_model.onnx"):
        continue

    src_path = os.path.join(SRC_DIR, filename)
    dst_path = os.path.join(DST_DIR, filename)

    if os.path.isfile(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"  - {filename}")


print("\n" + "=" * 60)
print(f"HOÀN TẤT. Model đã quantize nằm ở: {DST_DIR}")
print("Đổi MODEL_DIR trong app_onnx_manual.py sang thư mục này")
print("rồi chạy lại measure_ram.py để kiểm tra RAM mới.")
print("=" * 60)