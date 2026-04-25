import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# configuracion basica de la pagina
st.set_page_config(
    page_title="Asistente Starken",
    page_icon="📦",
    layout="centered"
)

# cargo variables del .env
load_dotenv()

# obtengo token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise ValueError("falta el GITHUB_TOKEN en el archivo .env")

# configuro entorno para usar github models
os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"

# funcion que inicializa todo (se ejecuta una vez)
@st.cache_resource
def inicializar_agente():

    # lista de documentos
    documentos = []

    # ruta donde estan los txt (misma carpeta)
    ruta_data = os.path.dirname(__file__)

    # cargar archivos txt
    for archivo in os.listdir(ruta_data):
        if archivo.endswith(".txt"):
            loader = TextLoader(os.path.join(ruta_data, archivo), encoding="utf-8")
            documentos.extend(loader.load())

    # dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documentos)

    # crear embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base="https://models.inference.ai.azure.com",
        openai_api_key=GITHUB_TOKEN
    )

    # guardar en chroma
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)

    # prompt del asistente
    prompt_template = """Eres un asistente virtual de Starken.
Responde SOLO con la informacion del contexto.
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

    # modelo de lenguaje
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_base="https://models.inference.ai.azure.com",
        openai_api_key=GITHUB_TOKEN
    )

    # retriever (buscador)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    return retriever, llm, PROMPT


# interfaz
st.title("📦 Asistente Virtual Starken")
st.caption("consulta sobre envios, tarifas y reclamos")

# historial
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# mostrar historial
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

# input usuario
if pregunta := st.chat_input("¿en que puedo ayudarte?"):

    # guardar pregunta
    st.session_state.mensajes.append({
        "rol": "user",
        "contenido": pregunta
    })

    with st.chat_message("user"):
        st.markdown(pregunta)

    # generar respuesta
    with st.chat_message("assistant"):
        with st.spinner("buscando informacion..."):

            retriever, llm, PROMPT = inicializar_agente()

            # 1 buscar documentos
            docs = retriever.invoke(pregunta)

            # 2 crear contexto
            contexto = "\n\n".join([doc.page_content for doc in docs])

            # 3 armar prompt
            prompt_final = PROMPT.format(
                context=contexto,
                question=pregunta
            )

            # 4 llamar modelo
            respuesta_llm = llm.invoke(prompt_final)
            respuesta = respuesta_llm.content

            # obtener fuentes
            fuentes = list(set([
                os.path.basename(doc.metadata.get("source", ""))
                for doc in docs
            ]))

            # mostrar respuesta
            st.markdown(respuesta)

            if fuentes:
                st.caption(f"fuentes: {', '.join(fuentes)}")

    # guardar respuesta
    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta
    })

