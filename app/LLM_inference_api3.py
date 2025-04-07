from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig
)
import requests
import torch
import json
import pandas as pd
import os
import time
from typing import Optional

MODEL_PATH = "model_trained/DeepSeek_R1_Distill_Qwen_14B/DeepSeek_R1_Distill_Qwen_14B_2025-03-28_14-25-20/model_merged"
os.environ['ROOT'] = "http://3.66.4.174:9090"

app = FastAPI()

class AnalyzeRequest(BaseModel):
    user_query: str
    data: Optional[dict] = None

DEFAULT_DATA_FILE = "DATA/mct_w120.json"

# Function to load local data
def load_default_data() -> dict:
    if os.path.exists(DEFAULT_DATA_FILE):
        with open(DEFAULT_DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise HTTPException(status_code=500, detail="No se encontró el archivo de datos por defecto.")

# Function to load Thingsboard's credentials
def load_credentials():
    with open('Credenciales.txt') as file:
        credenciales = file.read().strip()
    #print(credenciales)
    return credenciales

# Function to extract token from Thingsboard
def extract_token(credenciales):
    endpoint = f"{os.environ['ROOT']}/api/auth/login"
    headers = {"Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"}
    response = requests.post(endpoint, headers=headers, data=credenciales).json()
    return response['token'], response['refreshToken']

# Function to load device ID from Thingsboard
def extract_id(deviceName):
    endpoint = f"{os.environ['ROOT']}/api/tenant/devices?deviceName={str(deviceName)}"
    headers = {"Authorization": os.environ['TOKEN'],"Content-Type":"application/json;charset=UTF-8", "Accept":"application/json"}

    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        return response_json['id']['id']
    else:
        print(f"La página no existe. Codigo: {response.status_code}")
        return None

# Function to extract device's keys from Thingsboard
def extract_keys(deviceID):
    endpoint = f"{os.environ['ROOT']}/api/plugins/telemetry/DEVICE/{deviceID}/keys/timeseries"
    headers = {"Authorization": os.environ['TOKEN'], "Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"}

    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        response_json = response.json()
        return response_json
    else:
        print(f"Error al solicitar las keys. Codigo: {response.status_code}")
        return None

# Function to send a request to ThingsBoard, to get data from a device
def get_device_data(device_id):
    url = f"{os.environ['ROOT']}/api/v1/{device_id}/telemetry"

    headers = {'Authorization': os.environ['TOKEN']}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return None

# Function to extract telemetry data
def extract_telemetry(N, deviceID, keys):
    params = {
        "keys": keys,
        "startTs": 1000000000000,
        "endTs": 9999999999999,
        "limit": N,
        "orderBy": "DESC"
    }
    endpoint = f"{os.environ['ROOT']}/api/plugins/telemetry/DEVICE/{deviceID}/values/timeseries"
    headers = {"Authorization": os.environ['TOKEN'], "Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"}
    response = requests.get(endpoint, headers=headers, params=params)
    if response.status_code == 200:
        response_json = response.json()
        return response_json
    else:
        print(f"Error al extraer la telemetría: {response.status_code}")
        return None

def telemetry_to_df(raw_telemetry: dict, registros: int = 120) -> dict:
    """
    Transforma los datos raw extraídos desde ThingsBoard en formato de diccionario
    {índice: {param1: valor1, param2: valor2, ...}} con registros ordenados temporalmente.

    Args:
        raw_telemetry (dict): Telemetría obtenida desde ThingsBoard.
        registros (int): Número de registros (tiempos) a considerar.

    Returns:
        dict: Estructura igual a la del JSON de entrada de tu API.
    """
    # Convertimos a DataFrame intermedio para organizar por timestamp
    all_series = {}
    for param, entries in raw_telemetry.items():
        # Convertimos cada lista de dicts a una serie
        values = [float(e['value']) for e in entries]
        timestamps = [int(e['ts']) for e in entries]
        series = pd.Series(values, index=pd.to_datetime(timestamps, unit='ms'))
        all_series[param] = series
    # Unificamos todo en un DataFrame con índice temporal
    df = pd.DataFrame(all_series)
    df = df.sort_index()  # Asegura orden ascendente por tiempo
    # Nos quedamos con los N últimos registros (más recientes)
    df = df.tail(registros).reset_index(drop=True)
    return df.to_dict(orient="index")

def resumen_correlaciones(df: pd.DataFrame, umbral: float = 0.6) -> str:
    """
    Calcula la matriz de correlación y devuelve un resumen legible
    con pares de columnas cuya correlación es >= umbral o <= -umbral.

    Args:
        df (pd.DataFrame): DataFrame de entrada
        umbral (float): Umbral mínimo de correlación (por defecto 0.6)

    Returns:
        str: Texto con las correlaciones significativas
    """
    correlacion = df.corr()
    correlaciones_significativas = []

    # Iterar sobre pares únicos
    for col1 in correlacion.columns:
        for col2 in correlacion.columns:
            if col1 < col2:  # evita duplicados y diagonal
                valor = correlacion.loc[col1, col2]
                if abs(valor) >= umbral:
                    correlaciones_significativas.append(
                        f"{col1} ↔ {col2}: {round(valor, 3)}"
                    )

    if not correlaciones_significativas:
        return "No se encontraron correlaciones fuertes (con el umbral actual)."

    return "Correlaciones significativas (umbral >= " + str(umbral) + "):\n" + "\n".join(correlaciones_significativas)


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
                const payload = { "user_query": user_query};
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
@app.on_event("startup")
def load_model_on_startup():
    print("🔄 Cargando modelo y tokenizer...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True
    )
    app.state.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    app.state.tokenizer.pad_token = app.state.tokenizer.eos_token
    app.state.tokenizer.padding_side = "left"
    app.state.model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config
    )
    app.state.model.eval()
    print("✅ Modelo cargado y listo.")

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    print("📥 Petición recibida:")
    print("user_query:", request.user_query)
    print("data:", request.data)
    start_time = time.time()
    try:
        if request.data is not None:
            data = request.data
        else:
            print("📡 Cargando datos desde ThingsBoard...")
            # Load crendentials
            credenciales = load_credentials()
            # Extract token
            tok, _ = extract_token(credenciales)
            token = f"Bearer {tok}"
            os.environ['TOKEN'] = token
            # Puedes parametrizar el deviceName si lo necesitas desde el frontend
            device_id = extract_id("MCT Aggregation Mediana Ventana 1seg")  # o el que quieras
            keys = extract_keys(device_id)
            raw_telemetry = extract_telemetry(N=120, deviceID=device_id, keys=keys)
            data = telemetry_to_df(raw_telemetry)
        df = pd.DataFrame.from_dict(data, orient="index")
        # Tendencia
        n = 10
        early_avg = df.head(n).mean().round(3)
        late_avg = df.tail(n).mean().round(3)
        trend_summary = "\n".join([
            f"{col}: media_inicio={early_avg[col]}, media_final={late_avg[col]}"
            for col in df.columns
        ])
        # Métricas
        stats = df.describe()
        metrics_summary = "\n".join([
            f"{col}: min={round(stats.loc['min', col], 3)},"
            f" max={round(stats.loc['max', col], 3)},"
            f" mean={round(stats.loc['mean', col], 3)},"
            f" std={round(stats.loc['std', col], 3)}"
            for col in df.columns
        ])
        # Correlación
        correlation_text = resumen_correlaciones(df, umbral=0.25)
    except Exception as e:
        print("❌ Error procesando los datos:", e)
        raise HTTPException(status_code=400, detail=f"Error procesando los datos: {e}")

    prompt = (
        f"{request.user_query}\n\n"
        "**INSTRUCCIONES**\n"
        "- Tienes datos de una instalación con 51 parámetros.\n"
        "- Te voy a mostrar las métricas de dichos sensores.\n"
        "- No incluyas ni repeticiones del input ni la sección de </think>.\n"
        "- Responde en español.\n"
        "- La respuesta debe ser clara y estar bien estructurada.\n\n"
        f"**INFORMACIÓN DE LOS DATOS EXTRAÍDOS (media en los primeros y últimos {n} registros)**\n"
        f"{trend_summary}\n\n"
        f"**MÉTRICAS CALCULADAS**\n"
        f"{metrics_summary}\n\n"
        f"**MATRIZ DE CORRELACIÓN**\n"
        f"{correlation_text}\n\n"
        f"**RESPUESTA**\n"
    )

    inputs = app.state.tokenizer(prompt, return_tensors="pt").to(app.state.model.device)
    with torch.no_grad():
        output = app.state.model.generate(
            **inputs,
            max_new_tokens=5120,
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
    respuesta = app.state.tokenizer.decode(output[0], skip_special_tokens=True)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    print(f"⏱️ Tiempo de inferencia: {elapsed_time} segundos")
    return {"analysis": {"Model": respuesta}}