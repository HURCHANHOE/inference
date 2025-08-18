from huggingface_hub import hf_hub_download

model_name = "Qwen/Qwen3-8B-GGUF"
model_file = "Qwen3-8B-Q8_0.gguf"

model_path = hf_hub_download(model_name, filename=model_file, local_dir='./')
