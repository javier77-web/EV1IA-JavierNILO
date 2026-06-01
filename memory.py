"""
memory.py — Gestión de memoria del agente Starken

Memoria de corto plazo : ConversationBufferWindowMemory
    → Guarda los últimos N turnos de la conversación activa.
    → Se mantiene en RAM durante la sesión.

Memoria de largo plazo : ChromaDB persistente
    → Almacena resúmenes de conversaciones anteriores en disco.
    → Permite al agente recuperar contexto de sesiones pasadas.
"""

import os
import json
import datetime
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# CONFIGURACION
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

CHROMA_MEMORY_DIR = "./chroma_memory"   # directorio de memoria larga
CHROMA_KB_DIR     = "./chroma_db"       # directorio de base de conocimiento

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://models.inference.ai.azure.com",
    openai_api_key=GITHUB_TOKEN
)

# MEMORIA DE CORTO PLAZO
def crear_memoria_corto_plazo(k: int = 5):
    """
    Retorna un diccionario con historial de mensajes en memoria.
    k = número máximo de turnos a conservar.
    """
    return {"historial": [], "k": k}

# MEMORIA DE LARGO PLAZO (ChromaDB)
def obtener_vector_store_memoria() -> Chroma:
    """Devuelve el vector store de memoria persistente."""
    return Chroma(
        persist_directory=CHROMA_MEMORY_DIR,
        embedding_function=embeddings,
        collection_name="memoria_conversaciones"
    )


def guardar_en_memoria_larga(resumen: str, metadatos: dict = None):
    """
    Guarda un resumen de conversacion en ChromaDB persistente.
    Sirve para que el agente recuerde interacciones pasadas.
    """
    if not resumen or not resumen.strip():
        return

    vs = obtener_vector_store_memoria()

    meta = {
        "fecha": datetime.datetime.now().isoformat(),
        "tipo": "resumen_conversacion"
    }
    if metadatos:
        meta.update(metadatos)

    doc = Document(page_content=resumen, metadata=meta)
    vs.add_documents([doc])


def recuperar_contexto_largo_plazo(pregunta: str, k: int = 2) -> str:
    """
    Recupera los k resumenes de conversaciones pasadas
    mas relevantes semanticamente para la pregunta actual.
    Retorna un string con el contexto o vacio si no hay nada relevante.
    """
    try:
        vs = obtener_vector_store_memoria()
        collection = vs._collection
        if collection.count() == 0:
            return ""

        docs = vs.similarity_search(pregunta, k=k)
        if not docs:
            return ""

        fragmentos = []
        for doc in docs:
            fecha = doc.metadata.get("fecha", "fecha desconocida")[:10]
            fragmentos.append(f"[{fecha}] {doc.page_content}")

        return "Contexto de conversaciones anteriores:\n" + "\n".join(fragmentos)

    except Exception:
        return ""

# INDEXAR BASE DE CONOCIMIENTO (se llama una vez)
def inicializar_base_conocimiento(ruta_data: str) -> Chroma:
    """
    Carga los archivos .txt, los divide en chunks
    y los guarda en ChromaDB persistente.
    Si ya existe, simplemente lo devuelve.
    """
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # si ya existe la base, la cargamos directamente
    if os.path.exists(CHROMA_KB_DIR) and os.listdir(CHROMA_KB_DIR):
        return Chroma(
            persist_directory=CHROMA_KB_DIR,
            embedding_function=embeddings
        )

    # cargar archivos txt
    documentos = []
    for archivo in os.listdir(ruta_data):
        if archivo.endswith(".txt"):
            ruta = os.path.join(ruta_data, archivo)
            loader = TextLoader(ruta, encoding="utf-8")
            documentos.extend(loader.load())

    if not documentos:
        raise FileNotFoundError("No se encontraron archivos .txt en la ruta indicada.")

    # dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documentos)

    # guardar en chroma persistente
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_KB_DIR
    )

    return vector_store


# UTILIDAD: construir historial legible
def formatear_historial(mensajes: list) -> str:
    """
    Convierte la lista de mensajes de session_state
    en un string legible para incluir en el contexto del agente.
    """
    if not mensajes:
        return ""

    lineas = []
    for msg in mensajes[-6:]:   # ultimos 3 turnos
        rol = "Usuario" if msg["rol"] == "user" else "Asistente"
        lineas.append(f"{rol}: {msg['contenido']}")

    return "\n".join(lineas)