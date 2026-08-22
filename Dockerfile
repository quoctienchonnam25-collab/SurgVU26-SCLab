FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache Qwen2-VL-2B-Instruct model weights
# This is crucial because the evaluation platform does not have internet access.
RUN python -c "from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; \
    AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct'); \
    Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')"

# Copy src code and weights
COPY src /app/src
COPY checkpoints /app/checkpoints

# Set python path
ENV PYTHONPATH=/app

# The evaluation container runs with --network none (confirmed via the official
# category-2 reference container's do_test_run.sh) - force offline mode so
# huggingface_hub never attempts a revision-check network call at runtime.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Default command for Grand Challenge algorithm
# Grand Challenge mounts /input and expects outputs in /output
ENTRYPOINT ["python", "src/predict.py", "--input_dir", "/input", "--output_dir", "/output", "--model_path", "Qwen/Qwen2-VL-2B-Instruct", "--lora_path", "/app/checkpoints/qwen2_vl_2b_lora_v5"]
