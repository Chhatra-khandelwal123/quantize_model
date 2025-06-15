# quantize_model

# Quantize Model - 4-bit Quantization with BitsAndBytes

This project demonstrates how to load and run a **4-bit quantized Large Language Model (LLM)** using the `bitsandbytes` library and Hugging Face Transformers.

---

## Features

- Load LLM models in **4-bit precision** for reduced memory usage and faster inference.
- Supports double quantization and advanced quant types (`nf4`) for better accuracy.
- Automatic device placement (GPU/CPU) using `device_map="auto"`.
- Works with models like LLaMA 2, Falcon, and other compatible Hugging Face models.

---

## Requirements

- Python 3.8+
- CUDA-enabled GPU with at least 12GB VRAM (recommended)
- Packages:
  - `transformers`
  - `accelerate`
  - `bitsandbytes`

Install dependencies via pip:

```bash
pip install transformers accelerate bitsandbytes
