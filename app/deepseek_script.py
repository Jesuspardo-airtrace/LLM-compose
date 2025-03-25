import requests
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Function to load credentials
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

# Function to transform data to string
def transform_data_to_string(parameter, values):
    # Extraer los valores de cada parámetro
    param_values = [str(format(float(entry['value']), '.3f')) for entry in values]
    # Crear la línea con el formato deseado
    param_string = f"{parameter}: {', '.join(param_values)}"
    return param_string

# Function to make a promt from data loaded for DeepSeek
def analyze_telemetry(param, values):
    prompt = (f"Analiza estos datos de telemetría y genera un resumen: \n "
              f"Parámetro: {param} \n "
              f"Valores: {values} \n"
              f"Instrucciones: \n"
              f"- Identifica si los valores muestran estabilidad o variabilidad. \n"
              f"- Si hay alguna tendencia o patrón, descríbelo.\n"
              f"- Indica si existe algún valor fuera de lo esperado.\n"
              f"- Proporciona una conclusión sobre el comportamiento del parámetro.\n")
    #prompt = f"Hola."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Asegurar que los inputs están en GPU/CPU
    output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.5, pad_token_id=None)  # Sampling activado

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Imprimir debug
    print(f"🔹 Prompt enviado:\n{prompt}")
    print(f"🔹 Respuesta generada:\n{response}")

    return response

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load crendentials
credenciales = load_credentials()
# Extract token
tok, _ = extract_token(credenciales)
TOKEN = f"Bearer {tok}"
os.environ['TOKEN'] = TOKEN

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
data_est = extract_telemetry(N=12, deviceID=deviceID_est, keys=keysList_est)
print("Se han extraido los datos de estimaciones.")
data_gt = extract_telemetry(N=12, deviceID=deviceID_gt, keys=keysList_gt)
print("Se han extraido los datos de la captura.")

# DeepSeek model configuration
model_name = "deepseek-ai/deepseek-llm-7b-base"
model_path = "model/deepseek-llm-7b-base"
if not os.path.exists(f"/app/{model_path}"):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Model downloaded: {model_name.split('/')[-1]}.")
    # Guarda el modelo localmente
    model.save_pretrained(f"{model_path}")
    tokenizer.save_pretrained(f"{model_path}")
    print(f"Model saved in {model_path} folder.")
else:
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Model loaded successfully from {model_path}.")

if data_est:
    #for param, values in data_est.items():
    param, values = next(iter(data_est.items()))
    param_values = [str(format(float(entry['value']), '.3f')) for entry in values]
    values_str = ", ".join(map(str, param_values))
    data_est_txt_p = transform_data_to_string(parameter=param, values=values)
    print(f"estimaciones_txt param: {param}")
    print(f"estimaciones_txt values: {values_str}")
    print(f"estimaciones_txt values lenght: {len(values_str)}")
    summary = analyze_telemetry(param=param, values=values_str)
    print("🔹 Resumen de la telemetría de estimación:", summary)
