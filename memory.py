import os
import datetime
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

CHROMA_MEMORY_DIR = "./chroma_memory"
CHROMA_KB_DIR = "./chroma_db"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://models.inference.ai.azure.com",
    openai_api_key=GITHUB_TOKEN
)

def crear_memoria_corto_plazo(k: int = 5):
    return {"historial": [], "k": k}

def obtener_vector_store_memoria() -> Chroma:
    return Chroma(
        persist_directory=CHROMA_MEMORY_DIR,
        embedding_function=embeddings,
        collection_name="memoria_conversaciones"
    )

def guardar_en_memoria_larga(resumen: str, metadatos: dict = None):
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

def inicializar_base_conocimiento(ruta_data: str) -> Chroma:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if os.path.exists(CHROMA_KB_DIR) and os.listdir(CHROMA_KB_DIR):
        return Chroma(
            persist_directory=CHROMA_KB_DIR,
            embedding_function=embeddings
        )

    documentos = []
    for archivo in os.listdir(ruta_data):
        if archivo.endswith(".txt"):
            ruta = os.path.join(ruta_data, archivo)
            loader = TextLoader(ruta, encoding="utf-8")
            documentos.extend(loader.load())

    if not documentos:
        raise FileNotFoundError("No se encontraron archivos .txt en la ruta indicada.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documentos)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_KB_DIR
    )

    return vector_store

def formatear_historial(mensajes: list) -> str:
    if not mensajes:
        return ""

    lineas = []
    for msg in mensajes[-6:]:
        rol = "Usuario" if msg["rol"] == "user" else "Asistente"
        lineas.append(f"{rol}: {msg['contenido']}")

    return "\n".join(lineas)