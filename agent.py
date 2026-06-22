"""
agent.py — Agente funcional Starken (compatible LangChain 0.3.25)

Arquitectura:
    LCEL Agent (LangChain Expression Language)
        ├── LLM        : gpt-4o-mini via GitHub Models (bind_tools)
        ├── Tools      : buscar_informacion, calcular_tarifa, registrar_reclamo
        ├── Memory CP  : historial de mensajes en sesión
        ├── Memory LP  : ChromaDB persistente (sesiones anteriores)
        └── Prompt     : system prompt con rol + contexto histórico
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

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

# MAPA DE HERRAMIENTAS por nombre
TOOLS_MAP = {tool.name: tool for tool in TOOLS}


class AgenteStarken:
    """
    Agente compatible con LangChain 0.3.25 usando LCEL + bind_tools.
    Reemplaza AgentExecutor con un loop manual de tool-calling.
    """

    def __init__(self, contexto_largo_plazo: str = ""):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_base="https://models.inference.ai.azure.com",
            openai_api_key=GITHUB_TOKEN
        ).bind_tools(TOOLS)

        self.system_prompt = SYSTEM_PROMPT.format(
            contexto_largo_plazo=contexto_largo_plazo
        )

    def invoke(self, inputs: dict) -> dict:
        """
        Procesa una pregunta ejecutando el loop de razonamiento:
        LLM → tool calls → LLM → respuesta final.
        """
        pregunta = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])

        # Construir mensajes
        from langchain_core.messages import SystemMessage
        mensajes = [SystemMessage(content=self.system_prompt)]
        mensajes += chat_history
        mensajes.append(HumanMessage(content=pregunta))

        intermediate_steps = []
        max_iterations = 5

        for _ in range(max_iterations):
            respuesta = self.llm.invoke(mensajes)
            mensajes.append(respuesta)

            # Si no hay tool calls → respuesta final
            if not respuesta.tool_calls:
                break

            # Ejecutar cada herramienta solicitada
            for tc in respuesta.tool_calls:
                nombre = tc["name"]
                args   = tc["args"]
                tool_id = tc["id"]

                tool = TOOLS_MAP.get(nombre)
                if tool:
                    try:
                        resultado = tool.invoke(args)
                    except Exception as e:
                        resultado = f"[ERROR en {nombre}]: {e}"
                else:
                    resultado = f"[Herramienta '{nombre}' no encontrada]"

                intermediate_steps.append((nombre, resultado))
                mensajes.append(
                    ToolMessage(content=str(resultado), tool_call_id=tool_id)
                )

        output = respuesta.content if hasattr(respuesta, "content") else str(respuesta)

        return {
            "output": output,
            "intermediate_steps": intermediate_steps,
        }


# CONSTRUCCION DEL AGENTE
def crear_agente(contexto_largo_plazo: str = "") -> AgenteStarken:
    """Retorna una instancia del agente Starken."""
    return AgenteStarken(contexto_largo_plazo=contexto_largo_plazo)


# FUNCION PRINCIPAL: procesar pregunta
def procesar_pregunta(executor: AgenteStarken, pregunta: str) -> dict:
    """
    Procesa la pregunta del usuario con el agente.
    Retorna dict con: respuesta, herramientas_usadas, pasos_intermedios.
    """
    resultado = executor.invoke({"input": pregunta, "chat_history": []})

    respuesta = resultado.get("output", "No pude generar una respuesta.")

    # extraer herramientas usadas (sin duplicados)
    herramientas_usadas = []
    for nombre, _ in resultado.get("intermediate_steps", []):
        if nombre not in herramientas_usadas:
            herramientas_usadas.append(nombre)

    return {
        "respuesta": respuesta,
        "herramientas_usadas": herramientas_usadas,
        "pasos_intermedios": len(resultado.get("intermediate_steps", []))
    }


# MAIN — prueba en consola
if __name__ == "__main__":

    print("\n" + "="*60)
    print("  AGENTE STARKEN — Prueba en consola")
    print("="*60)

    ruta = os.path.dirname(__file__)
    print("\n[1/2] Inicializando base de conocimiento...")
    inicializar_base_conocimiento(ruta)
    print("[ok] Base de conocimiento lista.")

    print("[2/2] Creando agente...")
    executor = crear_agente()
    print("[ok] Agente listo.\n")

    casos_prueba = [
        {
            "pregunta": "Hola, necesito enviar un paquete de 3 kg a Punta Arenas a domicilio, ¿cuánto me cuesta?",
            "descripcion": "→ debe usar: calcular_tarifa"
        },
        {
            "pregunta": "¿Qué hago si mi paquete llegó dañado?",
            "descripcion": "→ debe usar: buscar_informacion + registrar_reclamo"
        },
        {
            "pregunta": "Mi paquete no ha llegado y ya pasaron 10 días desde que lo despacharon a Temuco",
            "descripcion": "→ debe usar: registrar_reclamo"
        },
        {
            "pregunta": "¿Tienen servicio express? Necesito que llegue mañana a Santiago",
            "descripcion": "→ debe usar: buscar_informacion"
        },
        {
            "pregunta": "Hace un momento me dijiste el precio para Punta Arenas, ¿ese precio incluye seguro?",
            "descripcion": "→ demuestra memoria de corto plazo"
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

    resumen = (
        "Sesión de prueba: cliente consultó sobre tarifa a Punta Arenas (3kg), "
        "paquete dañado, paquete perdido a Temuco, servicio express y seguros."
    )
    guardar_en_memoria_larga(resumen, {"sesion": "prueba_consola"})
    print("\n\n[ok] Resumen guardado en memoria de largo plazo.")
    print("="*60)