import os
import re
import datetime
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from observability import medir_latencia

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

os.environ["OPENAI_API_KEY"] = GITHUB_TOKEN
os.environ["OPENAI_API_BASE"] = "https://models.inference.ai.azure.com"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://models.inference.ai.azure.com",
    openai_api_key=GITHUB_TOKEN
)

INTENT_RULES = {
    "plazos": "plazos de entrega días hábiles regiones zonas extremas horarios atención",
    "express": "entrega rápida urgente prioritaria plazos de entrega horarios atención",
    "rápido": "entrega rápida urgente prioritaria plazos de entrega horarios atención",
    "urgente": "entrega rápida urgente prioritaria plazos de entrega horarios atención",
    "seguimiento": "tracking estado pedido seguimiento envío",
    "reclamo": "reclamo daño perdido entrega fallida devoluciones",
    "dañado": "paquete dañado reclamo devoluciones 48 horas",
    "perdido": "paquete perdido investigación seguro 15 días hábiles",
    "devolución": "política de devoluciones reenvío error remitente",
    "horarios": "horarios de atención call center sucursales web",
    "restricción": "restricciones de envío peso máximo dimensiones prohibido",
    "precio": "tarifa costo envío valor despacho",
}

def normalizar_consulta(pregunta: str) -> str:
    q = pregunta.lower()
    extras = []
    for clave, expansion in INTENT_RULES.items():
        if clave in q:
            extras.append(expansion)
    return f"{pregunta} {' '.join(extras)}".strip()

@medir_latencia(nombre_herramienta="buscar_informacion")
def _buscar_informacion_impl(pregunta: str) -> str:
    try:
        query = normalizar_consulta(pregunta)
        vector_store = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)

        if not docs:
            return (
                "INFORMACIÓN RECUPERADA DE LA BASE:\n\n"
                "No encontré coincidencias exactas, pero revisa plazos de entrega, horarios de atención, "
                "restricciones de envío y política de devoluciones."
            )

        contexto = "\n\n".join([
            f"[Fuente: {os.path.basename(doc.metadata.get('source', 'desconocida'))}]\n{doc.page_content}"
            for doc in docs
        ])

        return f"INFORMACIÓN RECUPERADA DE LA BASE:\n\n{contexto}"

    except Exception as e:
        return f"Error al buscar informacion: {str(e)}"

@tool
def buscar_informacion(pregunta: str) -> str:
    """
    Busca informacion sobre politicas de envio, seguimiento de paquetes,
    devoluciones, paquetes danados o perdidos, restricciones de envio y horarios de Starken.
    """
    return _buscar_informacion_impl(pregunta)

@medir_latencia(nombre_herramienta="calcular_tarifa")
def _calcular_tarifa_impl(descripcion: str) -> str:
    descripcion_lower = descripcion.lower()

    peso = None
    numeros = re.findall(r'(\d+(?:\.\d+)?)\s*(?:kg|kilo|kilos|gramos?|g\b)', descripcion_lower)
    unidades = re.findall(r'\d+(?:\.\d+)?\s*(kg|kilo|kilos|gramos?|g\b)', descripcion_lower)

    if numeros and unidades:
        valor = float(numeros[0])
        unidad = unidades[0]
        if 'g' in unidad and 'kg' not in unidad:
            peso = valor / 1000
        else:
            peso = valor

    if peso is None:
        return (
            "No pude detectar el peso del paquete. "
            "Por favor indica el peso, por ejemplo: '3 kg a Santiago' o '500g a Temuco'.\n\n"
            "Tarifas de referencia:\n"
            "- Sobre (hasta 500g): $2.990\n"
            "- Paquete pequeño (hasta 2 kg): $4.490\n"
            "- Paquete mediano (hasta 5 kg): $6.990\n"
            "- Paquete grande (hasta 10 kg): $9.990\n"
            "- Paquete extra grande (hasta 20 kg): $14.990\n"
            "- Carga (hasta 30 kg): $19.990"
        )

    if peso <= 0.5:
        tarifa_base = 2990
        tipo = "Sobre (hasta 500g)"
    elif peso <= 2:
        tarifa_base = 4490
        tipo = "Paquete pequeño (hasta 2 kg)"
    elif peso <= 5:
        tarifa_base = 6990
        tipo = "Paquete mediano (hasta 5 kg)"
    elif peso <= 10:
        tarifa_base = 9990
        tipo = "Paquete grande (hasta 10 kg)"
    elif peso <= 20:
        tarifa_base = 14990
        tipo = "Paquete extra grande (hasta 20 kg)"
    elif peso <= 30:
        tarifa_base = 19990
        tipo = "Carga (hasta 30 kg)"
    else:
        return f"El peso de {peso} kg supera el maximo de 30 kg por paquete. Para envios mayores se requiere servicio de carga especial."

    zonas_extremas = ["punta arenas", "magallanes", "aysen", "aysén", "arica", "parinacota"]
    zona_extrema = any(z in descripcion_lower for z in zonas_extremas)
    domicilio = any(p in descripcion_lower for p in ["domicilio", "casa", "a mi casa", "direccion"])
    express = any(p in descripcion_lower for p in ["express", "urgente", "rapido", "expreso"])

    total = tarifa_base
    desglose = [f"Tarifa base ({tipo}): ${tarifa_base:,}"]

    if zona_extrema:
        recargo_zona = int(tarifa_base * 0.40)
        total += recargo_zona
        desglose.append(f"Recargo zona extrema (+40%): +${recargo_zona:,}")

    if domicilio:
        total += 1500
        desglose.append("Entrega a domicilio: +$1.500")

    if express:
        recargo_express = int(tarifa_base * 0.60)
        total += recargo_express
        desglose.append(f"Recargo Starken Express (+60%): +${recargo_express:,}")

    desglose.append(f"\nTOTAL ESTIMADO: ${total:,} CLP")
    return "\n".join(desglose)

@tool
def calcular_tarifa(descripcion: str) -> str:
    """
    Calcula el costo estimado de un envio segun peso y destino.
    """
    return _calcular_tarifa_impl(descripcion)

@medir_latencia(nombre_herramienta="registrar_reclamo")
def _registrar_reclamo_impl(descripcion: str) -> str:
    descripcion_lower = descripcion.lower()

    if any(p in descripcion_lower for p in ["danado", "dañado", "roto", "golpeado", "deteriorado", "mal estado"]):
        tipo = "PAQUETE DAÑADO"
        instrucciones = (
            "PASOS A SEGUIR:\n"
            "1. Si el daño es visible al recibir: RECHAZA el paquete en el momento.\n"
            "2. Si notaste el daño al abrir: tienes 48 horas para reportarlo.\n"
            "3. Llama al 600 390 3000 con foto del daño y el embalaje.\n"
            "4. El reclamo se resuelve en un plazo máximo de 10 días hábiles.\n"
            "5. Formulario online: www.starken.cl/devoluciones"
        )
    elif any(p in descripcion_lower for p in ["perdido", "no llego", "no llegó", "no aparece", "desaparecio"]):
        tipo = "PAQUETE PERDIDO"
        instrucciones = (
            "PASOS A SEGUIR:\n"
            "1. Verifica que hayan pasado los plazos máximos de entrega.\n"
            "2. Llama al 600 390 3000 para iniciar investigación interna.\n"
            "3. Starken tiene hasta 15 días hábiles para investigar.\n"
            "4. Si se confirma la pérdida, se aplica seguro por valor declarado.\n"
            "5. Cobertura máxima sin seguro adicional: $50.000 CLP."
        )
    elif any(p in descripcion_lower for p in ["no estaba", "no habia nadie", "entrega fallida", "aviso", "reparto"]):
        tipo = "ENTREGA FALLIDA"
        instrucciones = (
            "PASOS A SEGUIR:\n"
            "1. El repartidor dejó un aviso en tu domicilio.\n"
            "2. Tienes 2 días hábiles para coordinar una nueva entrega.\n"
            "3. Llama al 600 390 3000 para reagendar.\n"
            "4. El costo de re-agendamiento es de $990 CLP.\n"
            "5. También puedes retirar el paquete en sucursal sin costo adicional."
        )
    elif any(p in descripcion_lower for p in ["cobro", "precio", "tarifa", "factura", "boleta", "caro"]):
        tipo = "PROBLEMA DE COBRO"
        instrucciones = (
            "PASOS A SEGUIR:\n"
            "1. Guarda el comprobante de pago original.\n"
            "2. Llama al 600 390 3000 y solicita revisión de tarifa.\n"
            "3. También puedes ir a una sucursal con el comprobante.\n"
            "4. Los reclamos de cobro se resuelven en hasta 5 días hábiles."
        )
    else:
        tipo = "RECLAMO GENERAL"
        instrucciones = (
            "CANALES DE ATENCIÓN:\n"
            "1. Call center: 600 390 3000 (lunes a sábado, 8:00 a 20:00).\n"
            "2. Formulario online: www.starken.cl/devoluciones\n"
            "3. Sucursales: atención presencial con tu número de tracking.\n"
            "4. Ten a mano: número de tracking, RUT y descripción del problema."
        )

    fecha = datetime.datetime.now().strftime("%d/%m/%Y %H:%M")

    return (
        f"RECLAMO REGISTRADO\n"
        f"Tipo: {tipo}\n"
        f"Fecha de registro: {fecha}\n\n"
        f"{instrucciones}\n\n"
        f"Guarda este registro como referencia para tu seguimiento."
    )

@tool
def registrar_reclamo(descripcion: str) -> str:
    """
    Registra un reclamo o incidencia del cliente y entrega instrucciones para resolverlo.
    """
    return _registrar_reclamo_impl(descripcion)

TOOLS = [buscar_informacion, calcular_tarifa, registrar_reclamo]