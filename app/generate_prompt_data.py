import pandas as pd
import numpy as np
import json

# ----------------------------------------------------------------
# 1. Carga del dataset (SIN columna datetime)
# ----------------------------------------------------------------

# Ajusta la ruta del CSV
csv_path = "app/DATA/data.csv"
df = pd.read_csv(csv_path)
# Observa las columnas que ha leído
print("\nColumnas detectadas:")
print(df.columns.tolist())
# Asegúrate de que todos los datos sean numéricos (o NaN si no pueden convertirse)
# (Si tu CSV ya contiene valores numéricos, en principio, no necesitas esto.
# Pero si temes strings raros, conviene asegurarse.)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# Opcional: verifica si hay NaN
nan_counts = df.isna().sum()
print("\nConteo de NaN por columna:")
print(nan_counts)


# ----------------------------------------------------------------
# 2. Función para crear PROMPT y RESPUESTA para cada ventana
# ----------------------------------------------------------------

def create_prompt_and_response(segment_df, window_index, param_columns):
    """
    Recibe un DataFrame con 120 filas y un índice de ventana.
    'param_columns' es la lista de nombres de columnas (51 en tu caso).

    Retorna un texto que combina un PROMPT solicitando el análisis
    y la RESPUESTA mostrando resultados estadísticos e interpretación.
    """

    # PROMPT
    prompt_text = (
        f"Tienes 120 registros pertenecientes a la ventana #{window_index}. "
        f"Las columnas son: {', '.join(param_columns)}.\n"
        "Por favor, realiza:\n"
        "1) Estadísticas por columna (min, max, mean, std)\n"
        "2) Detección de outliers (valores a ±3 std)\n"
        "3) Las correlaciones más altas (positivas)\n"
        "4) Una breve interpretación de los resultados\n"
    )

    # Cálculo de estadísticos
    stats = segment_df[param_columns].agg(["min", "max", "mean", "std"]).T

    # Detección de outliers
    outliers_count = {}
    for col in param_columns:
        mean_val = stats.loc[col, "mean"]
        std_val = stats.loc[col, "std"]
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val

        mask_outliers = (segment_df[col] < lower_bound) | (segment_df[col] > upper_bound)
        count_outl = mask_outliers.sum()
        if count_outl > 0:
            outliers_count[col] = int(count_outl)

    # Correlaciones
    corr_matrix = segment_df[param_columns].corr()
    corr_pairs = []
    for i in range(len(param_columns)):
        for j in range(i + 1, len(param_columns)):
            c1 = param_columns[i]
            c2 = param_columns[j]
            corr_val = corr_matrix.loc[c1, c2]
            corr_pairs.append((c1, c2, corr_val))

    # Ordena para tomar las correlaciones más altas (positivas)
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: x[2], reverse=True)
    top_3_corr = corr_pairs_sorted[:3]

    # Construir respuesta
    response_text = "Análisis de esta ventana (120 filas):\n\n"

    # 1) Estadísticas
    response_text += "1) Estadísticas:\n"
    for col in param_columns:
        mn = stats.loc[col, "min"]
        mx = stats.loc[col, "max"]
        mean_ = stats.loc[col, "mean"]
        std_ = stats.loc[col, "std"]
        response_text += (
            f"   - {col}: min={mn:.2f}, max={mx:.2f}, mean={mean_:.2f}, std={std_:.2f}\n"
        )

    # 2) Outliers
    response_text += "\n2) Outliers (±3 std):\n"
    if outliers_count:
        for k, v in outliers_count.items():
            response_text += f"   - {k}: {v} outliers\n"
    else:
        response_text += "   No se detectaron outliers en esta ventana.\n"

    # 3) Correlaciones
    response_text += "\n3) Análisis de outliers y sus correlaciones:\n"

    if not outliers_count:
        # Si no hubo columnas con outliers
        response_text += "   No se detectaron outliers en esta ventana.\n"
    else:
        # Mostramos las columnas con outliers, en orden
        response_text += "   Columnas que presentan outliers:\n"
        for col_out, count_val in outliers_count.items():
            response_text += f"      - {col_out} con {count_val} outliers\n"

        # Para cada columna con outliers, mostramos la gravedad y las correlaciones más altas
        response_text += "\n   Detalle de cada columna con outliers:\n"

        for col_out, count_val in outliers_count.items():
            response_text += f"   - {col_out}:\n"

            # == A) Cálculo de la 'gravedad' (usando desviación típica) ==
            mean_val = stats.loc[col_out, "mean"]
            std_val = stats.loc[col_out, "std"]

            # Obtenemos las filas concretas que son outliers en segment_df
            # (asumiendo la lógica ±3 std que ya calculaste)
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val

            # Filtramos la parte de segment_df donde col_out es outlier
            outlier_rows = segment_df[
                (segment_df[col_out] < lower_bound) | (segment_df[col_out] > upper_bound)
                ][col_out]

            # Determinamos la distancia máxima en std
            # Tomamos el valor que más se aleja de la media (arriba o abajo)
            if len(outlier_rows) > 0:
                # Calculamos |val - mean_val| / std_val y tomamos el máximo
                deviations = ((outlier_rows - mean_val).abs()) / std_val
                max_deviation = deviations.max()
                response_text += (
                    f"      * Se han detectado {count_val} outliers. "
                    f"El más extremo está a {max_deviation:.2f} std de la media.\n"
                )
            else:
                response_text += (
                    f"      * Se han detectado {count_val} outliers, "
                    "pero no se pudo calcular la desviación para el más extremo.\n"
                )

            # == B) Correlaciones con las otras columnas (|corr| >= 0.3) ==
            # Tomamos la serie de correlaciones de col_out
            corr_series = corr_matrix[col_out].drop(labels=[col_out])
            # Filtramos por valor absoluto >= 0.3 (ajusta si lo deseas)
            corr_series_filtered = corr_series[corr_series.abs() >= 0.3]
            # Ordenamos por valor absoluto desc
            corr_series_sorted = corr_series_filtered.abs().sort_values(ascending=False)
            # Tomamos las 3 más altas
            top_3_indices = corr_series_sorted.head(3).index

            if len(top_3_indices) == 0:
                response_text += "      * No hay parámetros con correlación >= 0.3.\n"
            else:
                response_text += "      * Parámetros más correlacionados (|corr| >= 0.3):\n"
                for col2 in top_3_indices:
                    real_corr = corr_matrix.loc[col_out, col2]
                    response_text += f"         -> {col2} (corr={real_corr:.3f})\n"

    # 4) Interpretación final
    total_outl = sum(outliers_count.values()) if outliers_count else 0
    response_text += (
        "\n4) Interpretación:\n"
        f"   En esta ventana, se detectaron {total_outl} outliers en total. "
        "Para cada columna con outliers, se ha calculado la distancia máxima "
        "en std con respecto a la media, lo que indica la gravedad del desvío. "
        "Además, se listan los parámetros con correlaciones superiores a ±0.3, "
        "que podrían influir o verse afectados en conjunto.\n"
    )

    # Unir PROMPT y RESPUESTA
    full_text = prompt_text + "\n\nRespuesta:\n" + response_text
    return full_text


# ----------------------------------------------------------------
# 3. Generar los ejemplos en ventanas de 120 filas
# ----------------------------------------------------------------

# Obtenemos la lista de columnas directamente de df
# (Todas son parámetros, en tu caso)
param_columns = df.columns.tolist()

WINDOW_SIZE = 120
all_examples = []
num_rows = len(df)
window_index = 0
start_idx = 0

while start_idx < num_rows:
    print(f"Procesando ventana {window_index}...")
    end_idx = start_idx + WINDOW_SIZE
    if end_idx > num_rows:
        break  # Evitar última ventana incompleta

    segment = df.iloc[start_idx:end_idx]

    # Genera prompt+respuesta
    example_text = create_prompt_and_response(segment, window_index, param_columns)
    # Almacenamos en una estructura para luego volcar a JSON
    example_data = {
        "window_index": window_index,
        "text": example_text
    }
    all_examples.append(example_data)

    start_idx += WINDOW_SIZE
    window_index += 1

print(f"\nSe generaron {len(all_examples)} ejemplos de entrenamiento.")


# ----------------------------------------------------------------
# 4. Guardar en formato JSON
# ----------------------------------------------------------------

output_json_path = "app/DATA/prompt_training_data.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)

print(f"\nEl archivo '{output_json_path}' se ha guardado con {len(all_examples)} ejemplos.")
