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

import os
import csv
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


# color palette
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"
# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput label {{
        color: black;
    }}
    .stError {{
        color: #FF0000; /* Rojo intenso */
    }}
    .stException {{
        color: #FF0000; /* Rojo intenso */
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# Crea un directorio para los archivos si no existe
pdf_dir = "/app/pdfs_data"
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)

# Streamlit app title
st.title("Build a RAG System with DeepSeek R1 & Ollama")

# Cargar el índice FAISS si existe, sino permitir cargar un nuevo archivo PDF
faiss_index_path = "/app/pdfs_data/faiss_index"

if os.path.exists(faiss_index_path):
    vector = FAISS.load_local(faiss_index_path, HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    st.write("FAISS index loaded successfully.")
else:
    vector = None
    st.write("No FAISS index found. Please upload a PDF file to create one.")


# Load the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    file_path = os.path.join(pdf_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.write(f"File saved at {file_path}")
    # Load the PDF
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # Split into chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    new_documents = text_splitter.split_documents(docs)

    if vector is not None:
        vector.add_documents(new_documents)
        st.write("New documents added to FAISS index.")
    else:
        vector = FAISS.from_documents(new_documents, HuggingFaceEmbeddings())
        st.write("New FAISS index created.")
    # Save new FAISS index
    vector.save_local(faiss_index_path)
    st.write("FAISS index updated and saved.")
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# REVSAR INDEXADOR
# Cargar el índice FAISS
vector = FAISS.load_local("/app/pdfs_data/faiss_index", HuggingFaceEmbeddings(), allow_dangerous_deserialization=True)

# Obtener los documentos indexados
docs = vector.docstore._dict

# Imprimir cantidad de documentos y una muestra
print(f"Total de documentos indexados: {len(docs)}")
for i, (key, doc) in enumerate(docs.items()):
    print(f"\nDocumento {i+1}:")
    print(f"ID: {key}")
    print(f"Contenido: {doc.page_content[:500]}...")  # Solo los primeros 500 caracteres
    print(f"Fuente: {doc.metadata.get('source', 'Desconocida')}")
    if i == 4:  # Mostrar solo los primeros 5 documentos
        break


# Define llm
llm = Ollama(model="deepseek-r1", base_url="http://ollama-service:11435")

# Define the prompt
prompt = """
Actúa como un experto en análisis de datos y en instalaciones de cualquier tipo.
1. Usa los siguientes fragmentos de contexto para responder a la pregunta del final.
2. Si no sabes la respuesta, solo di "No lo sé" pero no te inventes una respuesta por ti mismo.\n
3. Da una respuesta detallada.
4. Da la respuesta en español.
Aquí están los datos procesados de la ventana temporal de los últimos 15 minutos: 

{data}

Teniendo en cuenta estos datos, toma como contexto la información de todos los sensores de WADI, así como 
de los detalles técnicos de la instalación de WADI.

Contexto: {context}

Pregunta: {question}
"""

# Load processed data
with open(f"DATA/metrics_w10000.csv", mode='r', encoding='utf-8') as file:
    csv_content = file.read()
prompt = prompt.replace("{data}", csv_content)

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

llm_chain = LLMChain(
    llm=llm,
    prompt=QA_CHAIN_PROMPT,
    callbacks=None,
    verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="context",
    document_prompt=document_prompt,
    callbacks=None)

qa = RetrievalQA(
    combine_documents_chain=combine_documents_chain,
    verbose=True,
    retriever=retriever,
    return_source_documents=True)

# User input
user_input = st.text_input("Ask a question related to the PDF :")

# Process user input
if user_input:
    with st.spinner("Processing..."):
        response = qa(user_input)["result"]
        st.write("Response:")
        st.write(response)
        st.write(f"Prompt length: {len(user_input)}")
else:
    st.write("Please upload a PDF file to proceed.")