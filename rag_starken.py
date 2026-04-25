import os
from dotenv import load_dotenv  # para cargar variables del .env
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# cargo variables del archivo .env
load_dotenv()

# configuracion del token
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
# validacion basica para evitar error NoneType
if not GITHUB_TOKEN:
    raise ValueError("falta el GITHUB_TOKEN en el archivo .env")

# configuro entorno para usar github models
os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"


# paso 1: cargar documentos
def cargar_documentos():
    documentos = []

    ruta_data = os.path.join(os.path.dirname(__file__))

    for archivo in os.listdir(ruta_data):
        if archivo.endswith(".txt"):
            ruta = os.path.join(ruta_data, archivo)
            loader = TextLoader(ruta, encoding="utf-8")
            documentos.extend(loader.load())
            print(f"[ok] cargado: {archivo}")

    return documentos


# paso 2: dividir en chunks
def dividir_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documentos)

    print(f"[ok] total chunks: {len(chunks)}")

    return chunks


# paso 3: crear vector store
def crear_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base="https://models.inference.ai.azure.com",
        openai_api_key=GITHUB_TOKEN
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("[ok] vector store creado")

    return vector_store


# paso 4: crear modelo y prompt
def inicializar_modelo():

    prompt_template = """Eres un asistente virtual de Starken.
Responde solo con informacion del contexto.
Si no sabes responde: no tengo esa informacion.

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_base="https://models.inference.ai.azure.com",
        openai_api_key=GITHUB_TOKEN
    )

    return llm, PROMPT


# paso 5: hacer consulta (manual rag)
def consultar(vector_store, llm, PROMPT, pregunta):

    # 1. busco documentos relevantes
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(pregunta)

    # 2. armo contexto
    contexto = "\n\n".join([
        doc.page_content for doc in docs
    ])

    # 3. armo prompt final
    prompt_final = PROMPT.format(
        context=contexto,
        question=pregunta
    )

    # 4. envio al modelo
    respuesta_llm = llm.invoke(prompt_final)
    respuesta = respuesta_llm.content

    # imprimir resultado
    print("\n" + "="*50)
    print(f"pregunta: {pregunta}")
    print("-"*50)
    print("respuesta:")
    print(respuesta)
    print("-"*50)
    print("fuentes:")

    for doc in docs:
        fuente = doc.metadata.get("source", "desconocida")
        print(f"- {os.path.basename(fuente)}")

    print("="*50)

    return respuesta


# main para probar en consola
if __name__ == "__main__":

    print("\niniciando rag manual...\n")

    docs = cargar_documentos()

    chunks = dividir_chunks(docs)

    vs = crear_vector_store(chunks)

    llm, PROMPT = inicializar_modelo()

    print("\nsistema listo\n")

    preguntas = [
        "que hago si mi paquete llega dañado",
        "cuanto demora un envio",
    ]

    for p in preguntas:
        consultar(vs, llm, PROMPT, p)
        
        
# funcion para usar desde streamlit (frontend)
def inicializar_sistema():
    docs = cargar_documentos()
    chunks = dividir_chunks(docs)
    vs = crear_vector_store(chunks)
    llm, PROMPT = inicializar_modelo()

    return vs, llm, PROMPT


# funcion que recibe pregunta y devuelve respuesta (para app)
def preguntar(vs, llm, PROMPT, pregunta):
    return consultar(vs, llm, PROMPT, pregunta)