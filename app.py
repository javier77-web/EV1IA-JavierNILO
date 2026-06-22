"""
app.py — Interfaz web del Agente Starken
Conecta la UI Streamlit con el AgentExecutor de LangChain.
"""

import os
from dotenv import load_dotenv
import streamlit as st

# CONFIGURACION DE PAGINA (DEBE IR PRIMERO)
st.set_page_config(
    page_title="Asistente Starken",
    page_icon="📦",
    layout="centered"
)

# CARGAR VARIABLES DE ENTORNO
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    st.error("❌ Falta el GITHUB_TOKEN en el archivo .env")
    st.stop()

os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"

# ── EV3: OBSERVABILIDAD ──────────────────────────────────────────
from observability import MedidorEjecucion
# ─────────────────────────────────────────────────────────────────

# INICIALIZACION DEL AGENTE (una sola vez)
@st.cache_resource
def inicializar_sistema():
    """
    Inicializa la base de conocimiento y el agente.
    Se ejecuta solo una vez gracias a cache_resource.
    """
    from memory import inicializar_base_conocimiento, recuperar_contexto_largo_plazo
    from agent import crear_agente

    ruta_data = os.path.dirname(__file__)

    # 1. indexar documentos en ChromaDB
    inicializar_base_conocimiento(ruta_data)

    # 2. recuperar contexto de largo plazo (sesiones anteriores)
    contexto_lp = recuperar_contexto_largo_plazo("historial de conversaciones", k=2)

    # 3. crear agente con ese contexto
    executor = crear_agente(contexto_largo_plazo=contexto_lp)

    return executor


# ESTILOS CSS
st.markdown("""
<style>
    /* badge de herramienta */
    .tool-badge {
        display: inline-block;
        background: #e8f4f8;
        color: #1a6b8a;
        border: 1px solid #b3d9e8;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin-right: 4px;
        font-family: monospace;
    }
    /* contenedor de fuentes */
    .sources-container {
        background: #f8f9fa;
        border-left: 3px solid #e63946;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-top: 6px;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# INTERFAZ PRINCIPAL
st.title("📦 Asistente Virtual Starken")
st.caption("Agente inteligente · consultas, tarifas y reclamos")

# barra lateral: info del sistema 
with st.sidebar:
    st.markdown("### 🤖 Sobre este agente")
    st.markdown("""
**Herramientas disponibles:**
- 🔍 `buscar_informacion` — políticas y procedimientos
- 💰 `calcular_tarifa` — costos de envío
- 📋 `registrar_reclamo` — incidencias y problemas

**Memoria:**
- Corto plazo: últimos 5 turnos
- Largo plazo: ChromaDB persistente

**Modelo:** gpt-4o-mini (GitHub Models)
    """)

    st.divider()
    st.markdown("**Ejemplos de preguntas:**")
    ejemplos = [
        "¿Cuánto cuesta enviar 5kg a Arica?",
        "Mi paquete llegó dañado, ¿qué hago?",
        "¿Cuánto demora un envío a regiones?",
        "Quiero un envío express a Santiago",
        "Mi paquete no llega hace 10 días",
    ]
    for ej in ejemplos:
        if st.button(ej, use_container_width=True, key=f"ej_{ej[:15]}"):
            st.session_state["pregunta_rapida"] = ej

    st.divider()
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.session_state.pop("agente", None)
        st.cache_resource.clear()
        st.rerun()

# ESTADO DE SESION
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

if "agente" not in st.session_state:
    with st.spinner("⚙️ Inicializando agente..."):
        st.session_state.agente = inicializar_sistema()

# mostrar mensaje de bienvenida si no hay historial
if not st.session_state.mensajes:
    with st.chat_message("assistant"):
        st.markdown(
            "¡Hola! Soy el asistente virtual de **Starken** 📦\n\n"
            "Puedo ayudarte con:\n"
            "- 💰 Calcular el costo de tu envío\n"
            "- 🔍 Consultar plazos y políticas de envío\n"
            "- 📋 Registrar reclamos o incidencias\n\n"
            "¿En qué te puedo ayudar hoy?"
        )

# mostrar historial de mensajes
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])

        # mostrar metadatos si existen (herramientas usadas)
        if msg.get("herramientas"):
            badges = "".join(
                f'<span class="tool-badge">🔧 {t}</span>'
                for t in msg["herramientas"]
            )
            st.markdown(f'<div style="margin-top:4px">{badges}</div>', unsafe_allow_html=True)

# PROCESAR PREGUNTA
# manejar botones de ejemplo
pregunta_input = st.session_state.pop("pregunta_rapida", None)

# input del usuario
if pregunta_chat := st.chat_input("¿En qué puedo ayudarte?"):
    pregunta_input = pregunta_chat

if pregunta_input:

    # guardar y mostrar pregunta del usuario
    st.session_state.mensajes.append({
        "rol": "user",
        "contenido": pregunta_input
    })
    with st.chat_message("user"):
        st.markdown(pregunta_input)

    # generar respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Analizando y buscando información..."):

            from agent import procesar_pregunta
            from memory import guardar_en_memoria_larga, formatear_historial

            # ── EV3: MEDICIÓN DE OBSERVABILIDAD ─────────────────
            with MedidorEjecucion(pregunta=pregunta_input, herramienta="agente_completo") as m:
                try:
                    resultado = procesar_pregunta(
                        st.session_state.agente,
                        pregunta_input
                    )
                    respuesta = resultado["respuesta"]
                    herramientas = resultado["herramientas_usadas"]
                    m.set_respuesta(respuesta)
                except Exception as e:
                    respuesta = f"⚠️ Error al procesar la consulta: {e}"
                    herramientas = []
                    m.set_error(str(e))
            # ─────────────────────────────────────────────────────

            # mostrar respuesta
            st.markdown(respuesta)

            # mostrar badges de herramientas usadas
            if herramientas:
                badges = "".join(
                    f'<span class="tool-badge"> {t}</span>'
                    for t in herramientas
                )
                st.markdown(
                    f'<div style="margin-top:6px">{badges}</div>',
                    unsafe_allow_html=True
                )

    # guardar respuesta en historial
    st.session_state.mensajes.append({
        "rol": "assistant",
        "contenido": respuesta,
        "herramientas": herramientas
    })

    # cada 5 intercambios, guardar resumen en memoria larga
    if len(st.session_state.mensajes) % 10 == 0:
        historial_texto = formatear_historial(st.session_state.mensajes[-10:])
        resumen = f"Conversación con usuario: {historial_texto[:500]}"
        guardar_en_memoria_larga(resumen, {"origen": "streamlit"})