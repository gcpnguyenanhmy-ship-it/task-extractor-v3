# FROM ghcr.io/ggml-org/llama.cpp:server

# COPY model/task_extractor_smollm2_v1_Q4_K_M.gguf /model/model.gguf

# ENV PORT=10000

# EXPOSE 10000

# ENTRYPOINT ["llama-server"]

# CMD [
#     "--model", "/model/model.gguf",
#     "--host", "0.0.0.0",
#     "--port", "10000",
#     "--ctx-size", "512",
#     "--n-predict", "80",
#     "--threads", "2",
#     "--parallel", "1",
#     "--gpu-layers", "0"
# ]
# ============================================================
# Dockerfile - Task Extractor ONNX API
#
# Dùng python:3.11-slim để tối ưu dung lượng image + RAM.
# KHÔNG cài torch/optimum - chỉ dùng onnxruntime thuần để
# giảm RAM tối đa, phù hợp Render free tier (512MB RAM).
# ============================================================

FROM python:3.11-slim

WORKDIR /app

# --------------------------------------------------------
# Cài dependencies trước để tận dụng Docker layer cache
# (chỉ rebuild lại bước này khi requirements.txt thay đổi)
# --------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------
# Copy code + model ONNX đã convert sẵn
#
# LƯU Ý: model_onnx/ phải đã có sẵn trong thư mục project
# (được tạo bằng lệnh optimum-cli export onnx) TRƯỚC khi
# build Docker image - không convert lúc runtime.
# --------------------------------------------------------

COPY app_onnx_manual.py .
COPY model_onnx/ ./model_onnx/

# --------------------------------------------------------
# Render cung cấp PORT qua biến môi trường $PORT
# (không cố định port cứng như 8000)
# --------------------------------------------------------

ENV PORT=8000

EXPOSE 8000

CMD uvicorn app_onnx_manual:app --host 0.0.0.0 --port ${PORT}