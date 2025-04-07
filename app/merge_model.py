import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from huggingface_hub import login, snapshot_download

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
print("¿Token presente?:", bool(token))
login(token=token)

def merge_finetuned_model(base_model_dir, finetuned_model_dir: str, merged_output_dir: str):
    """
    Carga un modelo fine-tuneado con adaptadores desde 'finetuned_model_dir',
    realiza la fusión (merge) de los pesos del adaptador en el modelo base,
    y guarda el modelo completo (merge) en 'merged_output_dir'.
    """
    if not os.path.exists(base_model_dir):
        print(f"Descargando modelo base '{model_name}' a '{base_model_dir}'...")
        os.makedirs(base_model_dir, exist_ok=True)

        # Descarga el modelo en la carpeta indicada
        snapshot_download(
            repo_id=model_name,
            local_dir=base_model_dir,
            local_dir_use_symlinks=False  # Copia archivos reales, no enlaces simbólicos
        )
    offload_dir = os.path.join(base_model_dir, "offload")
    os.makedirs(offload_dir, exist_ok=True)
    os.environ["ACCELERATE_OFFLOAD_DIR"] = offload_dir
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        "device_map": "auto",
        "offload_folder": offload_dir
    }
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        low_cpu_mem_usage=True,
        **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir,
        trust_remote_code=True)
    # Cargar el adaptador (finetuning) sobre el modelo base
    print("Cargando adaptador finetuneado desde:", finetuned_model_dir)
    peft_model = PeftModel.from_pretrained(
        base_model,
        finetuned_model_dir,
        device_map="auto",
        offload_folder=offload_dir)

    # Fusionar (merge) los pesos del adaptador en el modelo base y descargar el adaptador
    print("Fusionando adaptador en el modelo base...")
    merged_model = peft_model.merge_and_unload()

    # Guardar el modelo fusionado y el tokenizer en la carpeta de salida
    print("Guardando el modelo fusionado en:", merged_output_dir)
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    print("¡Modelo fusionado guardado correctamente!")
    #### #### #### #### ####

if __name__ == "__main__":
    ###########################################
    # 1. Descargar o cargar el modelo base
    ###########################################
    # Selecciona el modelo base que deseas utilizar.
    # Para LLaMA-2 13B (modo chat):
    ##model_name = "meta-llama/Llama-2-13b-chat-hf"
    # Para Mistral 7B, podría ser algo como:
    model_name = "mistralai/Mistral-7B-v0.1"
    # Para Deepseek LLM 7B:
    ##model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    model_name_0 = model_name.split('/')[-1].replace("-", "_")
    # Ajusta estas rutas según tu estructura
    base_model_dir = os.path.join("model", model_name.replace("/", "_"))
    finetuned_model_dir = "model_trained/Mistral_7B_v0.1/Mistral_7B_v0.1_2025-03-27_15-14-13/checkpoint-1182"
    merged_output_dir = "model_trained/Mistral_7B_v0.1/Mistral_7B_v0.1_2025-03-27_15-14-13/model_merged_fp16"

    merge_finetuned_model(base_model_dir, finetuned_model_dir, merged_output_dir)
