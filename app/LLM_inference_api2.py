from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from safetensors.torch import load_file
from collections import Counter
import uvicorn
import os
import time




##### Verificación #####
def is_quantized_safetensors(safetensor_path):
    try:
        state_dict = load_file(safetensor_path)
        dtype_counter = Counter(tensor.dtype for tensor in state_dict.values())
        is_quantized = any(dtype in [torch.int8, torch.uint8] for dtype in dtype_counter)
        return is_quantized, dtype_counter
    except Exception as e:
        print(f"Error al leer {safetensor_path}: {e}")
        return None, None

def scan_models(base_dir="model_trained"):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".safetensors") and "model" in file:
                full_path = os.path.join(root, file)
                model_name = root.split("model_trained/")[-1]
                result, dtypes = is_quantized_safetensors(full_path)
                if result is True:
                    print(f"[✓] {model_name} → CUANTIZADO (4-bit)")
                elif result is False:
                    print(f"[x] {model_name} → NO cuantizado (fp16 o fp32)")
                else:
                    print(f"[!] {model_name} → No se pudo verificar")
                if dtypes:
                    print(f"    Tipos encontrados: {dict(dtypes)}")

#scan_models()

app = FastAPI(
    title="API de Evaluación de Modelos Fine-Tuneados",
    description="Calcula métricas a partir de un JSON de datos y evalúa modelos fine-tuneados utilizando un prompt construido a partir de dichas métricas.",
    version="1.0"
)

##############################################
# 1. Definir rutas de los modelos fine-tuneados
##############################################
# Cada ruta debe apuntar a la carpeta donde se encuentran los archivos del modelo (p.ej., pytorch_model.bin, tokenizer.json, etc.)
model_dirs = {
    "Model": "model_trained/DeepSeek_R1_Distill_Qwen_14B/DeepSeek_R1_Distill_Qwen_14B_2025-03-28_14-25-20/model_merged",
    #"Model": "model_trained/Llama_2_13b_chat_hf/Llama_2_13b_chat_hf_2025-03-31_07-12-38/model_merged_fp16",
    #"Model": "model_trained/Mistral_7B_v0.1/Mistral_7B_v0.1_2025-03-27_15-14-13/model_merged_fp16"
}
MODEL_PATH = model_dirs['Model']


##############################################
# 2. Definir el esquema del request
##############################################
class AnalyzeRequest(BaseModel):
    user_query: str
    # 'data' es opcional; si no se proporciona, se utilizarán datos predeterminados
    data: dict = None

# Archivo de datos predeterminado
DEFAULT_DATA_FILE = "DATA/mct_w120.json"

def load_default_data() -> dict:
    if os.path.exists(DEFAULT_DATA_FILE):
        with open(DEFAULT_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=500, detail="No se encontró el archivo de datos por defecto.")


##############################################
# 3. Funciones para cálculo de métricas y generación del prompt
##############################################
def calculate_metrics_from_json(data: dict) -> str:
    """
    Calcula min, max, mean y std por columna y devuelve el resumen como string formateado.
    """
    try:
        df = pd.DataFrame.from_dict(data, orient="index")
    except Exception as e:
        raise ValueError(f"Error al convertir JSON a DataFrame: {e}")

    stats = df.describe()
    summary_lines = []

    for col in df.columns:
        try:
            mn = round(stats.loc["min", col], 3)
            mx = round(stats.loc["max", col], 3)
            mean_ = round(stats.loc["mean", col], 3)
            std_ = round(stats.loc["std", col], 3)
            summary_lines.append(
                f"{col}: min={mn}, max={mx}, mean={mean_}, std={std_}"
            )
        except Exception as e:
            summary_lines.append(f"{col}: Error calculando métricas ({e})")

    return "\n".join(summary_lines)


def build_prompt(user_query: str, metrics_summary: str) -> str:
    base_query = (
        "**INSTRUCCIONES**\n"
        "- Tienes datos de una instalación con 51 parámetros.\n"
        "- Te voy a mostrar las métricas de dichos sensores.\n"
        "- No incluyas ni repeticiones del input ni la sección de </think>.\n"
        "- Responde en español.\n"
        "- La respuesta debe ser clara y estar bien estructurada."
    )
    prompt = f"{user_query}\n\n{base_query}\n\nMétricas calculadas:\n{metrics_summary}\n\nRespuesta:\n"
    return prompt


##############################################
# 4. Función para evaluar un modelo dado un prompt
##############################################
def evaluate_model(model_dir: str, prompt_text: str) -> str:
    # Inicio
    start_time = time.time()
    try:
        # Detectar si hay cuantización válida bitsandbytes
        safetensor_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        is_valid_quant = False
        for f in safetensor_files:
            result, _ = is_quantized_safetensors(os.path.join(model_dir, f))
            if result:
                is_valid_quant = True
                break
        # Usar kwargs apropiados
        if is_valid_quant:
            model_args = {                          # Carga en 4-bit
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True   # ✅ AÑADE esto
                ),
                "device_map": "auto"
            }
            print("Cargando modelo como cuantizado (4-bit).")
        else:
            model_args = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                "device_map": "auto"
            }
            print("Cargando modelo como no cuantizado (fp16/bf16).")

        # Cargar la configuración y forzar un valor por defecto a model_type si es necesario
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        if not hasattr(config, "model_type") or config.model_type is None:
            # Forzamos un valor; aquí usamos "gpt2" como ejemplo. Ajusta si es necesario.
            config.model_type = "gpt2"
            print(f"Se ha forzado config.model_type a 'gpt2' para el modelo en {model_dir}.")
        # Cargar el tokenizer y el modelo usando la configuración modificada
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, **model_args)
        except Exception as e:
            model_dir_offload = os.path.dirname(model_dir)          # En caso de no disponer de suficiente memoria
            print("Fallo al cargar en 4-bit, reintentando en fp16...")
            fallback_args = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
                "device_map": "sequential",
                "low_cpu_mem_usage": True,
                "offload_folder": model_dir_offload
            }
            model = AutoModelForCausalLM.from_pretrained(model_dir, **fallback_args)
        #model = AutoModelForCausalLM.from_pretrained(model_dir, **model_args)
        print(f"Modelo cargado: {model_dir}")
        print(f"Primera capa dtype: {next(model.parameters()).dtype}")
    except Exception as e:
        raise RuntimeError(f"Error al cargar el modelo desde {model_dir}: {e}")

    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=5120,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    return f"{response}\n\n⏱️ Tiempo de inferencia: {elapsed_time} segundos"


##############################################
# 5. Endpoint GET para la interfaz web
##############################################
@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Evaluación de Modelos Fine-Tuneados</title>
    </head>
    <body>
        <h1>Evaluación de Modelos Fine-Tuneados</h1>
        <form id="queryForm">
            <label for="user_query">Ingresa tu consulta:</label><br>
            <textarea id="user_query" name="user_query" rows="4" cols="50"></textarea><br><br>
            <input type="submit" value="Enviar">
        </form>
        <h2>Respuesta:</h2>
        <pre id="responseArea" style="white-space: pre-wrap;"></pre>
        <script>
            const form = document.getElementById('queryForm');
            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                const user_query = document.getElementById('user_query').value;
                const payload = { "user_query": user_query };
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();
                const rawText = data.analysis?.Model || "No se encontró análisis";
                document.getElementById('responseArea').textContent = rawText;
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

##############################################
# 6. Endpoint POST para análisis
##############################################
@app.post("/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    data = request.data if request.data is not None else load_default_data()
    try:
        metrics_summary = calculate_metrics_from_json(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    prompt = build_prompt(request.user_query, metrics_summary)
    print("Prompt de evaluación construido:")
    print(prompt)

    results = {}
    for model_name_key, model_dir in model_dirs.items():
        try:
            response = evaluate_model(model_dir, prompt)
            results[model_name_key] = response
        except Exception as e:
            results[model_name_key] = f"Error: {e}"
    return {"analysis": results}


##############################################
# 7. Despliegue de la API
##############################################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
