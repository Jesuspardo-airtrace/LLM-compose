import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.schema import Document

import os
import csv
import json
import subprocess
import requests

def check_ollama_connection(endpoint):
    try:
        response = requests.get(endpoint)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False

if not check_ollama_connection("http://ollama-service:11435"):
    st.error("Ollama is not reachable. Please make sure it is running.")
else:
    pass

# ============ PDF index stuff =============
pdf_dir = "/app/pdfs_data"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

st.title("Build a RAG System with DeepSeek R1 & Ollama")

faiss_index_path = "/app/pdfs_data/faiss_index"

# 1) Si hay un index
if os.path.exists(faiss_index_path):
    vector = FAISS.load_local(faiss_index_path, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    st.write("FAISS index loaded successfully.")
else:
    vector = None
    st.write("No FAISS index found. Please upload a PDF file to create one.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    file_path = os.path.join(pdf_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.write(f"File saved at {file_path}")

    # Procesar PDF
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    new_documents = text_splitter.split_documents(docs)

    if vector is not None:
        vector.add_documents(new_documents)
        st.write("New documents added to FAISS index.")
    else:
        vector = FAISS.from_documents(new_documents, HuggingFaceEmbeddings())
        st.write("New FAISS index created.")

    # Save
    vector.save_local(faiss_index_path)
    st.write("FAISS index updated and saved.")

# Cargar / recargar
if vector:
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# ============ LLM config =============
llm = Ollama(model="deepseek-r1", base_url="http://ollama-service:11435")

# ============ JSON data (no indexing) =============
with open("DATA/metrics_w10000.json","r", encoding="utf-8") as f:
    datos_json = json.load(f)

# ============ UTILS =============
def get_pdf_reference(sensor_name: str):
    """
    Usa el retriever para buscar 'sensor_name' en el PDF y devolver la info
    """
    if not vector:
        return "No hay PDF indexado."
    docs = retriever.get_relevant_documents(sensor_name)
    # Combinar o elegir el chunk mas relevante
    if not docs:
        return "No se encontró ese sensor en el PDF."
    # Podrías concatenar si quieres
    combined_text = "\n".join([d.page_content for d in docs])
    return combined_text

def get_json_actual(sensor_name: str):
    """
    Accede a datos_json. Asume que sensor_name esta EXACTO en las keys.
    """
    return datos_json.get(sensor_name, {})

# ============ Prompt para comparacion =============
compare_prompt = """
*** Reglas ***
1. Sensor {sensor_name_reference} es el sensor de referencia (PDF).
2. Sensor {sensor_name_actual} es el sensor con datos actuales (JSON).
3. No inventes datos. Si no aparece la métrica en el PDF o en el JSON, di "No lo sé".
4. Da la respuesta en español, explicando si hay diferencias notables.
5. Cada sensor es independiente; no mezclar con otros.
6. Devuelve un análisis breve.

*** Valores PDF (Referencia) para {sensor_name_reference} ***
{pdf_text}

*** Valores JSON (Actual) para {sensor_name_actual} ***
{json_text}

Pregunta: {question}
"""

compare_template = PromptTemplate(
    template=compare_prompt,
    input_variables=["sensor_name_reference","sensor_name_actual","pdf_text","json_text","question"]
)

def compare_sensors(sensor_name_reference, sensor_name_actual, question="¿Cuál es la diferencia?"):
    pdf_text = get_pdf_reference(sensor_name_reference)
    data_actual = get_json_actual(sensor_name_actual)
    if not data_actual:
        jtxt = "No lo sé, no hay datos en el JSON."
    else:
        jtxt = json.dumps(data_actual, indent=2)

    final_prompt = compare_template.format(
        sensor_name_reference=sensor_name_reference,
        sensor_name_actual=sensor_name_actual,
        pdf_text=pdf_text,
        json_text=jtxt,
        question=question
    )
    return llm(final_prompt)


# ============ MAIN UI =============
question = st.text_input("Pregunta:")

if question:
    # Ejemplo de parse. Si detectas "actual" => usas JSON, etc.
    if "actual" in question.lower():
        # CASO 1: el user quiere "datos actuales" => extrae sensor
        # Simplificado:
        sensor_name = "1_AIT_001_PV_actual"
        # Solo un ejemplo. En la vida real, parsear el question para saber cual sensor
        data = get_json_actual(sensor_name)
        if not data:
            st.write("No lo sé, no hay datos actuales para ese sensor.")
        else:
            st.write(data)
    elif "compara" in question.lower():
        # CASO 2: el user pide comparacion
        # supongamos q se parseo "1_AIT_001_PV_reference" vs "1_AIT_001_PV_actual"
        sensor_ref = "1_AIT_001_PV_reference"
        sensor_act = "1_AIT_001_PV_actual"
        ans = compare_sensors(sensor_ref, sensor_act, question)
        st.write(ans)
    else:
        # CASO 3: normal retrieval from PDF
        # e.g. "¿Qué dice el PDF sobre la turbidez?"
        if not vector:
            st.write("No hay PDF indexado.")
        else:
            # use retriever-based QA
            # Build a simpler prompt
            base_prompt = """
            *** Reglas ***
            1. Usa SOLO el PDF. No hay JSON en este caso.
            2. Si no lo encuentras, di "No lo sé".
            3. Responde en español.

            Contexto: {context}
            Pregunta: {question}
            """
            baseTemplate = PromptTemplate(input_variables=["context","question"], template=base_prompt)

            docChain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=baseTemplate),
                document_variable_name="context",
            )
            qachain = RetrievalQA(
                combine_documents_chain=docChain,
                retriever=retriever
            )
            response = qachain.run(question)
            st.write(response)
