import time
import numpy as np
import json
from collections import Counter

import requests
import os
import torch

# Environment variables
os.environ['ROOT'] = "http://3.66.4.174:9090"


# Function to load credentials
def load_credentials():
    with open('app/Credenciales.txt') as file:
        credenciales = file.read().strip()
    #print(credenciales)
    return credenciales

# Function to extract token from Thingsboard
def extract_token(credenciales):
    endpoint = f"{os.environ['ROOT']}/api/auth/login"
    headers = {"Content-Type": "application/json;charset=UTF-8", "Accept": "application/json"}
    response = requests.post(endpoint, headers=headers, data=credenciales).json()
    return response['token'], response['refreshToken']

# Function to load device ID
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

# Function to extract device's keys
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

# Function to extract telemetry data
def extract_telemetry(N, deviceID, keys):
    params = {
        "keys": keys,
        "startTs": 100000000000,
        "endTs": 99999999999999,
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

# Function to make a prompt from data loaded for DeepSeek
def analyze_telemetry(prompt_text, OLLAMA_URL):
    prompt = prompt_text

    # Datos que se envían a la API
    data = {
        #"model": "deepseek-r1",       # Nombre del modelo cargado en Ollama
        #"model": "deepseek-r1:14b",  # Deepseek de 14b parámetros
        "model": "deepseek-r1:32b",  # Deepseek de 32b parámetros
        "prompt": prompt,             # Pregunta que quieres enviar
        "stream": False               # Si lo pones en True, recibe la respuesta en partes (útil para chatbots)
    }
    # Hacer la solicitud POST a la API de Ollama
    response = requests.post(OLLAMA_URL, json=data)
    if response.status_code == 200:
        result = response.json()
        #print("\n💡 Respuesta del modelo:\n")
        #print(result["response"])  # Aquí está la respuesta del modelo
        return result
    else:
        print("❌ Error en la solicitud:", response.status_code, response.text)

def statistics(param_values):
    # 1. Valor mínimo y tiempo en minutos que se ha mantenido en torno a ese valor
    v_min = min(param_values)
    intervalo_minimo = (v_min, v_min * 1.1)
    tiempo_minimo = sum(1 for valor in param_values if intervalo_minimo[0] <= valor <= intervalo_minimo[1])
    t_v_min = (tiempo_minimo*8) /60
    # 2. Valor máximo y tiempo en minutos que se ha mantenido en torno a ese valor
    v_max = max(param_values)
    intervalo_maximo = (v_max * 0.9, v_max)
    tiempo_maximo = sum(1 for valor in param_values if intervalo_maximo[0] <= valor <= intervalo_maximo[1])
    t_v_max = (tiempo_maximo * 8) / 60
    # 3. Moda, media y desviación estándar
    v_mode = Counter(param_values).most_common(1)[0][0]  # Moda
    v_mean = np.mean(param_values)  # Media
    v_sdt = np.std(param_values)  # Desviación estándar
    # 4. Outliers
    q1 = np.percentile(param_values, 25)
    q3 = np.percentile(param_values, 75)
    iqr = q3 - q1
    v_out = param_values[(param_values < q1 - 1.5 * iqr) | (param_values > q3 + 1.5 * iqr)]

    return (np.round(v_min,5),
            np.round(t_v_min,5),
            np.round(v_max,5),
            np.round(t_v_max,5),
            np.round(v_mean,5),
            np.round(v_mode,5),
            np.round(v_sdt,5),
            np.round(v_out,5))


device = "cuda" if torch.cuda.is_available() else "cpu"

time_0 = time.time()

# Load crendentials
credenciales = load_credentials()
# Extract token
tok, _ = extract_token(credenciales)
TOKEN = f"Bearer {tok}"
os.environ['TOKEN'] = TOKEN
time_1 = time.time()
print(f"Tiempo para extraer token: {round(time_1-time_0, 2)} segundos.")

# Extract device ID
deviceID_est = extract_id("MCT Estimaciones M2")
deviceID_gt = extract_id("MCT Aggregation Mediana Ventana 1seg")

# Extract available keys
keysList_est = extract_keys(deviceID_est)
keysString_est = ','.join(keysList_est)
keysList_gt = extract_keys(deviceID_gt)
keysString_gt = ','.join(keysList_gt)

# Extract telemetry data
print("=== Extracting telemetry ===")
data_est = extract_telemetry(N=120, deviceID=deviceID_est, keys=keysList_est)
print("Se han extraido los datos de estimaciones.")
data_gt = extract_telemetry(N=120, deviceID=deviceID_gt, keys=keysList_gt)
print("Se han extraido los datos de la captura.")
time_2 = time.time()
print(f"Tiempo para extraer datos: {round(time_2-time_1, 2)} segundos.")

# URL de Ollama en local
OLLAMA_URL = "http://localhost:11434/api/generate"

# Load template.txt
with open("app/template/prompt_4_2.txt", "r", encoding="utf-8") as file:
    prompt_text = file.read()

prompts = []  # Lista para almacenar cada prompt por parámetro
if data_gt:
    param_f = list(data_gt.keys())[1:]
    for index, (param, values) in enumerate(data_gt.items()):
        if index == 0:
            continue  # Salta el primer elemento que es el timestamp
        param_values = [(format(float(entry['value']), '.5f')) for entry in values]
        param_values = np.array(param_values, dtype=float)
        v_min, t_v_min, v_max, t_v_max, v_mean, v_mode, v_sdt, v_out = statistics(param_values=param_values)
        with open("app/template/prompt_4_1.txt", "r", encoding="utf-8") as file:
            prompt_text_param = file.read()
        prompt_dict = json.loads(prompt_text_param)
        # Formatear el prompt para este parámetro
        prompt_param = {
            "parameter": param,
            "min_val": v_min,
            "t_v_min": t_v_min,
            "max_val": v_max,
            "t_v_max": t_v_max,
            "mean_val": v_mean,
            "mode_val": v_mode,
            "std_val": v_sdt,
            "outliers": v_out.tolist()
        }
        prompt_string = json.dumps(prompt_param, indent=2, ensure_ascii=False)
        prompts.append(prompt_string)  # Agregar el prompt a la lista

    # Unir todos los prompts en un solo string para enviarlo a la LLM
    full_prompt = (f"Eres un experto en análisis de series temporales y diagnóstico técnico. A continuación, te proporcionaré una lista de parámetros con sus métricas precalculadas. Tu tarea es analizarlos e identificar patrones críticos.\n"
                   f"**Lista de parámetros**:  {param_f}\n"
                   f"Cada parámetro incluye:  \n"
                   f"1. Nombre \n"
                   f"2. Métricas: \n"
                   f" - Mínimo \n"
                   f" - Máximo \n"
                   f" - Media \n"
                   f" - Moda \n"
                   f" - Desviación estándar (σ) \n"
                   f" - Tiempo en torno al mínimo (min) \n"
                   f" - Tiempo en torno al máximo (min) \n"
                   f" - Outliers detectados \n"+ "\n".join(prompts) + prompt_text)
    print(f"Longitud del prompt: {len(full_prompt)}")
    with open("app/output/prompt_4.txt", "w", encoding="utf-8") as file:
        file.write(full_prompt)
    summary = analyze_telemetry(prompt_text=full_prompt, OLLAMA_URL=OLLAMA_URL)
    print("🔹 Resumen de la telemetría de estimación:", summary['response'])
    time_3 = time.time()
    print(f"Tiempo para obtener respuesta del LLM: {round(time_3 - time_2, 2)} segundos.")

