#!/bin/bash

# Iniciar Ollama en segundo plano
ollama serve &

# Esperar a que Ollama esté listo (más tiempo si es necesario)
sleep 10

# Intentar descargar el modelo si no está ya presente
if ! ollama list | grep -q "deepseek-r1"; then
  echo "Modelo no encontrado. Descargando el modelo deepseek-r1..."
  ollama pull deepseek-r1
else
  echo "El modelo deepseek-r1 ya está descargado."
fi

# Terminar el proceso de Ollama una vez descargado
kill %1
