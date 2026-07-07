import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from tools import TOOLS
from memory import inicializar_base_conocimiento

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Falta GITHUB_TOKEN en el archivo .env")

os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"

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
4. Para cualquier consulta sobre políticas, horarios, plazos, seguimiento o procedimientos → usa buscar_informacion.
5. Si una pregunta no está relacionada con Starken, responde amablemente que solo puedes ayudar con temas de la empresa.
6. Si buscar_informacion entrega contexto relevante, responde usando exclusivamente esa información.
7. Si el contexto no es suficiente, dilo brevemente y pide el dato faltante.
8. Si la herramienta devuelve información útil sobre plazos, horarios o restricciones, no digas que no existe información.

{contexto_largo_plazo}
"""

TOOLS_MAP = {tool.name: tool for tool in TOOLS}

class AgenteStarken:
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
        pregunta = inputs.get("input", "")
        chat_history = inputs.get("chat_history", [])

        mensajes = [SystemMessage(content=self.system_prompt)]
        mensajes += chat_history
        mensajes.append(HumanMessage(content=pregunta))

        intermediate_steps = []
        max_iterations = 5
        ultima_respuesta = None

        for _ in range(max_iterations):
            respuesta = self.llm.invoke(mensajes)
            ultima_respuesta = respuesta
            mensajes.append(respuesta)

            if not respuesta.tool_calls:
                break

            for tc in respuesta.tool_calls:
                nombre = tc["name"]
                args = tc["args"]
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
                mensajes.append(ToolMessage(content=str(resultado), tool_call_id=tool_id))

        output = ultima_respuesta.content if ultima_respuesta and hasattr(ultima_respuesta, "content") else ""
        output_final = output

        if any(nombre == "buscar_informacion" for nombre, _ in intermediate_steps):
            contexto_recuperado = "\n\n".join(
                str(resultado) for nombre, resultado in intermediate_steps if nombre == "buscar_informacion"
            )
            if contexto_recuperado and "No encontré coincidencias exactas" not in contexto_recuperado:
                mensajes.append(HumanMessage(
                    content=(
                        "Redacta una respuesta final breve, clara y directa usando exclusivamente "
                        "la información entregada por buscar_informacion. No digas que no tienes datos."
                    )
                ))
                respuesta_final = self.llm.invoke(mensajes)
                output_final = respuesta_final.content if hasattr(respuesta_final, "content") else str(respuesta_final)

        return {
            "output": output_final,
            "intermediate_steps": intermediate_steps,
        }

def crear_agente(contexto_largo_plazo: str = ""):
    return AgenteStarken(contexto_largo_plazo=contexto_largo_plazo)

def procesar_pregunta(executor: AgenteStarken, pregunta: str, chat_history=None) -> dict:
    resultado = executor.invoke({
        "input": pregunta,
        "chat_history": chat_history or []
    })

    respuesta = resultado.get("output", "No pude generar una respuesta.")

    herramientas_usadas = []
    for nombre, _ in resultado.get("intermediate_steps", []):
        if nombre not in herramientas_usadas:
            herramientas_usadas.append(nombre)

    return {
        "respuesta": respuesta,
        "herramientas_usadas": herramientas_usadas,
        "pasos_intermedios": len(resultado.get("intermediate_steps", []))
    }

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AGENTE STARKEN — Prueba en consola")
    print("=" * 60)

    ruta = os.path.dirname(__file__)
    print("\n[1/2] Inicializando base de conocimiento...")
    inicializar_base_conocimiento(ruta)
    print("[ok] Base de conocimiento lista.")

    print("[2/2] Creando agente...")
    executor = crear_agente()
    print("[ok] Agente listo.\n")

    casos_prueba = [
        "¿Tienen servicio express? Necesito que llegue mañana a Santiago",
        "¿Qué hago si mi paquete llegó dañado?",
        "¿Cuánto cuesta enviar 3 kg a Punta Arenas a domicilio?",
    ]

    for i, pregunta in enumerate(casos_prueba, 1):
        print(f"\n{'─' * 60}")
        print(f"CASO {i}")
        print(f"Pregunta: {pregunta}")
        print("─" * 60)

        resultado = procesar_pregunta(executor, pregunta)
        print(f"\nRESPUESTA:\n{resultado['respuesta']}")
        print(f"\nHerramientas usadas: {resultado['herramientas_usadas']}")
        print(f"Pasos de razonamiento: {resultado['pasos_intermedios']}")