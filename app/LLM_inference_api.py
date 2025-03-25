from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn

###########################################
# 1. Configuración y carga del modelo fine-tuneado
###########################################

# Suponemos que ya realizaste el fine-tuning y el modelo se guardó en "output_llm_timeseries"
model_path = "./output_llm_timeseries"  # Ajusta la ruta según corresponda

print(f"Cargando modelo fine-tuneado desde: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


###########################################
# 2. Definición de la clase de request
###########################################

class AnalyzeRequest(BaseModel):
    data: list[list[float]]  # Cada sublista es una fila de datos
    columns: list[str]  # Lista de nombres de columnas (ej. 51 parámetros)
    user_query: str  # La pregunta o consulta que realiza el usuario


###########################################
# 3. Funciones para cálculo de métricas y generación de análisis
###########################################

def calculate_metrics_on_data(df: pd.DataFrame) -> str:
    """
    Calcula estadísticas descriptivas globales, detecta outliers (±3 std) y
    obtiene las top 3 correlaciones (por valor absoluto) del DataFrame.
    Retorna un string con el resumen de métricas.
    """
    # Estadísticas globales
    stats = df.describe().to_string()

    # Detección de outliers
    outliers_info = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        col_mean = df[col].mean()
        col_std = df[col].std()
        lower_bound = col_mean - 3 * col_std
        upper_bound = col_mean + 3 * col_std
        mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = int(mask.sum())
        if n_outliers > 0:
            outliers_info.append(f"{col}: {n_outliers} outliers")
    outliers_str = "\n".join(outliers_info) if outliers_info else "No se detectaron outliers (±3 std)."

    # Correlaciones: se toman todos los pares (i < j) y se ordenan por valor absoluto
    corr_matrix = df.corr().abs()
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            pairs.append((cols[i], cols[j], val))
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)
    top_3 = pairs_sorted[:3]
    correl_str = "\n".join([f"{a} - {b} (corr={val:.3f})" for (a, b, val) in
                            top_3]) if top_3 else "No se encontraron correlaciones destacadas."

    metrics_text = (
        "=== ANÁLISIS DE MÉTRICAS ===\n\n"
        f"Estadísticas descriptivas globales:\n{stats}\n\n"
        f"Outliers detectados:\n{outliers_str}\n\n"
        f"Correlaciones más fuertes (valor absoluto):\n{correl_str}\n"
    )
    return metrics_text


def generate_analysis_from_data(df: pd.DataFrame, user_query: str) -> str:
    """
    Combina:
    1) Cálculo de métricas en tiempo real (stats, outliers, correlaciones).
    2) Construcción de un prompt que inyecta esos resultados junto con la consulta del usuario.
    3) Llamada al modelo fine-tuneado para generar el análisis final.
    """
    # Calcular las métricas
    metrics_results = calculate_metrics_on_data(df)

    # Construir el prompt de entrada para el LLM
    prompt = (
        f"{user_query}\n\n"
        "A continuación tienes un resumen estadístico de los datos:\n"
        f"{metrics_results}\n"
        "Basándote en esta información, por favor elabora un análisis detallado, "
        "incluyendo interpretaciones sobre tendencias, estabilidad, outliers y correlaciones."
        "\n\nRespuesta:\n"
    )

    # Tokenizar y generar respuesta con el modelo fine-tuneado
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_length=500,  # Ajusta según el tamaño de respuesta esperado
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.7
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


###########################################
# 4. Creación de la API con FastAPI
###########################################

app = FastAPI(
    title="API de Análisis de Datos con LLM Fine-Tuneado",
    description="Recibe datos, calcula métricas (estadísticas, outliers, correlaciones) y genera un análisis basado en un modelo fine-tuneado.",
    version="1.0"
)


@app.post("/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    try:
        # Crear DataFrame a partir de la data y los nombres de columnas recibidos
        df_input = pd.DataFrame(request.data, columns=request.columns)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al crear DataFrame: {str(e)}")

    # Generar análisis utilizando la función definida
    analysis_result = generate_analysis_from_data(df_input, request.user_query)
    return {"analysis": analysis_result}


###########################################
# 5. Despliegue de la API
###########################################

if __name__ == "__main__":
    # Desplegar en host 0.0.0.0 y puerto 8000 (ajusta si es necesario)
    uvicorn.run(app, host="0.0.0.0", port=8000)
