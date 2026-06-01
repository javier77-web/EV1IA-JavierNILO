"""
agent.py — Agente funcional Starken

Arquitectura:
    LangChain AgentExecutor
        ├── LLM        : gpt-4o-mini via GitHub Models
        ├── Tools      : buscar_informacion, calcular_tarifa, registrar_reclamo
        ├── Memory CP  : ConversationBufferWindowMemory (ultimos 5 turnos)
        ├── Memory LP  : ChromaDB persistente (sesiones anteriores)
        └── Prompt     : system prompt con rol + contexto historico
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import TOOLS
from memory import (
    crear_memoria_corto_plazo,
    guardar_en_memoria_larga,
    recuperar_contexto_largo_plazo,
    inicializar_base_conocimiento,
)

# CONFIGURACION
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Falta GITHUB_TOKEN en el archivo .env")

os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"


# PROMPT SYSTEM DEL AGENTE
SYSTEM_PROMPT = """Eres el asistente virtual inteligente de Starken, empresa líder de courier en Chile.

Tu misión es ayudar a los clientes con:
- Consultas sobre envíos, plazos y seguimiento de paquetes
- Cálculo de tarifas y costos de despacho
- Registro y orientación sobre reclamos (paquetes dañados, perdidos, entregas fallidas)
- Información sobre políticas de envío y restricciones

REGLAS DE COMPORTAMIENTO:
1. Siempre usa las herramientas disponibles antes de responder. No inventes información.
2. Si el usuario pregunta por un costo o tarifa → usa la herramienta calcular_tarifa.
3. Si el usuario reporta un problema o reclamo → usa registrar_reclamo.
4. Para cualquier consulta sobre políticas o procedimientos → usa buscar_informacion.
5. Si una pregunta no está relacionada con Starken, responde amablemente que solo puedes ayudar con temas de la empresa.
6. Mantén un tono profesional pero cercano. Usa el contexto de la conversación para dar respuestas coherentes.

{contexto_largo_plazo}
"""

# CONSTRUCCION DEL AGENTE
def crear_agente(contexto_largo_plazo: str = ""):
    """
    Construye y retorna el AgentExecutor con todas las herramientas
    y la memoria de corto plazo configurada.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_base="https://models.inference.ai.azure.com",
        openai_api_key=GITHUB_TOKEN
    )

    # prompt con soporte para historial de mensajes y agent_scratchpad
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT.format(
            contexto_largo_plazo=contexto_largo_plazo
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # crear agente con OpenAI tools
    agent = create_openai_tools_agent(
        llm=llm,
        tools=TOOLS,
        prompt=prompt
    )

    # memoria de corto plazo
    memoria = crear_memoria_corto_plazo(k=5)

     # executor: combina agente + herramientas
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    return executor

# FUNCION PRINCIPAL: procesar pregunta
def procesar_pregunta(executor: AgentExecutor, pregunta: str) -> dict:
    """
    Procesa la pregunta del usuario con el agente.
    Retorna dict con: respuesta, herramientas_usadas, pasos_intermedios.
    """
    resultado = executor.invoke({"input": pregunta, "chat_history": []})

    respuesta = resultado.get("output", "No pude generar una respuesta.")

    # extraer herramientas usadas
    herramientas_usadas = []
    pasos = resultado.get("intermediate_steps", [])
    for accion, _ in pasos:
        nombre_tool = getattr(accion, "tool", None)
        if nombre_tool and nombre_tool not in herramientas_usadas:
            herramientas_usadas.append(nombre_tool)

    return {
        "respuesta": respuesta,
        "herramientas_usadas": herramientas_usadas,
        "pasos_intermedios": len(pasos)
    }

# MAIN — prueba en consola
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  AGENTE STARKEN — Prueba en consola")
    print("="*60)

    # inicializar base de conocimiento
    ruta = os.path.dirname(__file__)
    print("\n[1/2] Inicializando base de conocimiento...")
    inicializar_base_conocimiento(ruta)
    print("[ok] Base de conocimiento lista.")

    # crear agente
    print("[2/2] Creando agente...")
    executor = crear_agente()
    print("[ok] Agente listo.\n")

    # casos de prueba que demuestran toma de decisiones
    casos_prueba = [
        {
            "pregunta": "Hola, necesito enviar un paquete de 3 kg a Punta Arenas a domicilio, ¿cuánto me cuesta?",
            "descripcion": "→ debe usar: calcular_tarifa (zona extrema + domicilio)"
        },
        {
            "pregunta": "¿Qué hago si mi paquete llegó dañado?",
            "descripcion": "→ debe usar: buscar_informacion + registrar_reclamo"
        },
        {
            "pregunta": "Mi paquete no ha llegado y ya pasaron 10 días desde que lo despacharon a Temuco",
            "descripcion": "→ debe usar: registrar_reclamo (paquete perdido)"
        },
        {
            "pregunta": "¿Tienen servicio express? Necesito que llegue mañana a Santiago",
            "descripcion": "→ debe usar: buscar_informacion"
        },
        {
            "pregunta": "Hace un momento me dijiste el precio para Punta Arenas, ¿ese precio incluye seguro?",
            "descripcion": "→ demuestra memoria de corto plazo (recuerda la primera pregunta)"
        },
    ]

    for i, caso in enumerate(casos_prueba, 1):
        print(f"\n{'─'*60}")
        print(f"CASO {i}: {caso['descripcion']}")
        print(f"Pregunta: {caso['pregunta']}")
        print("─"*60)

        resultado = procesar_pregunta(executor, caso["pregunta"])

        print(f"\nRESPUESTA:\n{resultado['respuesta']}")
        print(f"\nHerramientas usadas: {resultado['herramientas_usadas']}")
        print(f"Pasos de razonamiento: {resultado['pasos_intermedios']}")

    # guardar resumen de esta sesion en memoria larga
    resumen = (
        "Sesión de prueba: cliente consultó sobre tarifa a Punta Arenas (3kg), "
        "paquete dañado, paquete perdido a Temuco, servicio express y seguros."
    )
    guardar_en_memoria_larga(resumen, {"sesion": "prueba_consola"})
    print("\n\n[ok] Resumen guardado en memoria de largo plazo.")
    print("="*60)