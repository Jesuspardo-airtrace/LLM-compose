import os
import json
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

###########################################
# 1. Descargar o cargar el modelo base
###########################################
# Selecciona el modelo base que deseas utilizar.
# Ejemplo para LLaMA-2 13B (modo chat):
##model_name = "meta-llama/Llama-2-13b-chat-hf"
# Para Mistral 7B, podría ser algo como:
#model_name = "mistralai/Mistral-7B-v0.1"
# Ejemplo para Deepseek LLM 7B:
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name_0 = model_name.split('/')[-1].replace("-","_")

# Define una carpeta local donde almacenar el modelo base
base_model_dir = os.path.join("model", model_name.replace("/", "_"))

token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Modificación: Añadir parámetros específicos para Deepseek
if "deepseek" in model_name.lower():
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    }
    tokenizer_kwargs = {
        "trust_remote_code": True
    }
else:
    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True
    }
    tokenizer_kwargs = {}

#quant_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_compute_dtype=torch.bfloat16)

# Si la carpeta no existe, descarga el modelo (se guardará en cache_dir)
if not os.path.exists(base_model_dir):
    print(f"Descargando modelo base '{model_name}' a '{base_model_dir}'...")
    if "deepseek" in model_name.lower() or "qwen" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
            #quantization_config=quant_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            pad_token="<|endoftext|>"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     cache_dir=base_model_dir,
                                                     token=token,
                                                     low_cpu_mem_usage=True,
                                                     **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=base_model_dir, token=token,
        **tokenizer_kwargs)
else:
    print(f"Cargando modelo base desde la carpeta local '{base_model_dir}'...")
    if "deepseek" in model_name.lower() or "qwen" in model_name.lower():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
            #quantization_config=quant_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            pad_token="<|endoftext|>"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            low_cpu_mem_usage=True,
            **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_dir,
            **tokenizer_kwargs)

# Algunos modelos (como LLaMA-2) no tienen pad_token definido
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Deepseek requiere ajustes especiales en el tokenizer
if "deepseek" in model_name.lower():
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% for message in messages %}{{message['content']}}{% endfor %}"
elif tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Mover a dispositivo adecuado (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model.to(device)
model.eval()
model.gradient_checkpointing_enable()      # permite recomputar algunos intermedios en lugar de almacenarlos

###########################################
# 2. Cargar el dataset de entrenamiento desde JSON
###########################################
json_path = "DATA/prompt_training_data.json"
dataset = load_dataset("json", data_files={"data": json_path})
# Dividir el dataset en train y eval (por ejemplo, 90%-10%)
dataset = dataset["data"].train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

###########################################
# 3. Tokenización
###########################################
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",  # O "longest" según tu preferencia
        truncation=True,
        max_length=512         # Ajusta según la longitud promedio de tus prompts
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

###########################################
# 4. Callback para guardar métricas y fecha de entrenamiento
###########################################
os.makedirs(f"model_trained/{model_name_0}", exist_ok=True)
class MetricsLoggingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        # Recopilamos logs de entrenamiento y la fecha actual
        training_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics = state.log_history  # Lista de logs durante el entrenamiento
        output = {
            "training_date": training_date,
            "metrics": metrics
        }
        metrics_filename = f"model_trained/{model_name_0}/{model_name_0}_metrics_{training_date}.json"
        with open(metrics_filename, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Métricas de entrenamiento guardadas en {metrics_filename}")

###########################################
# 5. Configuración del entrenamiento
###########################################
batch_size = 1      # Ajusta según la GPU
epochs = 3          # Ajusta según tus necesidades

# El directorio de salida incluirá la fecha de entrenamiento
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"model_trained/{model_name_0}/{model_name_0}_{current_date}"

bf16_supported = torch.cuda.is_bf16_supported()

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=8,  # Simula un batch size mayor
    num_train_epochs=epochs,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=not bf16_supported,  # Si no se soporta bf16, usa fp16
    bf16=bf16_supported,      # Solo uno puede ser True
    optim="adamw_torch_fused",
    load_best_model_at_end=True,
    report_to=["none"],  # Para evitar reportes a WandB u otros, opcional
    gradient_checkpointing=True  # Ahorro de memoria crítico
)

#data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # mlm=False para modelos de lenguaje causal
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[MetricsLoggingCallback()]
)

###########################################
# 6. Iniciar el entrenamiento
###########################################
print("Iniciando entrenamiento...")
trainer.train()

###########################################
# 7. Guardar el modelo fine-tuneado
###########################################
# El modelo se guarda automáticamente en el directorio output_dir,
# pero aquí se puede realizar una copia o notificar.
print(f"Modelo fine-tuneado guardado en: {output_dir}")
