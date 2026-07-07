"""
observability.py — Módulo de Observabilidad para Agente Starken
EV3 - ISY0101 Ingeniería de Soluciones con IA
"""

import time
import json
import os
import uuid
from datetime import datetime
from functools import wraps

# ─── Configuración ───────────────────────────────────────────────
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "metricas.jsonl")

os.makedirs(LOG_DIR, exist_ok=True)

# ─── Registro de una ejecución ───────────────────────────────────
def registrar_ejecucion(
    pregunta: str,
    respuesta: str,
    herramienta_usada: str,
    latencia_seg: float,
    error: bool = False,
    detalle_error: str = None,
    tokens_aprox: int = None,
):
    """
    Escribe una línea JSON en metricas.jsonl con todos los datos
    de una ejecución del agente.
    """
    registro = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "pregunta": pregunta,
        "respuesta": respuesta[:200] if respuesta else "",   # truncar para no inflar el log
        "herramienta": herramienta_usada,
        "latencia_seg": round(latencia_seg, 3),
        "error": error,
        "detalle_error": detalle_error,
        "tokens_aprox": tokens_aprox if tokens_aprox else len(respuesta.split()) if respuesta else 0,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro

# ─── Decorador para medir latencia de herramientas ───────────────
def medir_latencia(nombre_herramienta: str):
    """
    Decorador que envuelve una tool de LangChain y registra su latencia.
    Uso:
        @medir_latencia("buscar_informacion")
        def buscar_informacion(query):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            inicio = time.time()
            error = False
            detalle_error = None
            resultado = ""
            try:
                resultado = func(*args, **kwargs)
            except Exception as e:
                error = True
                detalle_error = str(e)
                resultado = f"[ERROR] {e}"
                raise
            finally:
                latencia = time.time() - inicio
                pregunta = args[0] if args else kwargs.get("query", "")
                registrar_ejecucion(
                    pregunta=str(pregunta),
                    respuesta=str(resultado),
                    herramienta_usada=nombre_herramienta,
                    latencia_seg=latencia,
                    error=error,
                    detalle_error=detalle_error,
                )
            return resultado
        return wrapper
    return decorator

# ─── Medición manual (para usar en app.py alrededor del agente) ──
class MedidorEjecucion:
    """
    Contexto para medir una llamada completa al agente.

    Uso en app.py:
        with MedidorEjecucion(pregunta=user_input) as m:
            respuesta = agent_executor.invoke({"input": user_input, ...})
            m.set_respuesta(respuesta["output"])
    """
    def __init__(self, pregunta: str, herramienta: str = "agente_completo"):
        self.pregunta = pregunta
        self.herramienta = herramienta
        self.inicio = None
        self._respuesta = ""
        self._error = False
        self._detalle_error = None

    def __enter__(self):
        self.inicio = time.time()
        return self

    def set_respuesta(self, texto: str):
        self._respuesta = texto

    def set_error(self, detalle: str):
        self._error = True
        self._detalle_error = detalle

    def __exit__(self, exc_type, exc_val, exc_tb):
        latencia = time.time() - self.inicio
        if exc_type:
            self._error = True
            self._detalle_error = str(exc_val)
        registrar_ejecucion(
            pregunta=self.pregunta,
            respuesta=self._respuesta,
            herramienta_usada=self.herramienta,
            latencia_seg=latencia,
            error=self._error,
            detalle_error=self._detalle_error,
        )
        return False  # no suprimir excepciones

# ─── Carga y resumen de métricas ─────────────────────────────────
def cargar_logs() -> list[dict]:
    """Retorna todos los registros del archivo JSONL como lista de dicts."""
    if not os.path.exists(LOG_FILE):
        return []
    registros = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    registros.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return registros

def calcular_metricas_resumen(registros: list[dict]) -> dict:
    """
    Calcula métricas agregadas sobre todos los registros.
    Retorna un dict con:
      - total_ejecuciones
      - tasa_error (%)
      - latencia_promedio_seg
      - latencia_max_seg
      - latencia_min_seg
      - latencia_por_herramienta (dict)
      - precisión_proxy (% ejecuciones sin error)
      - consistencia_proxy (% preguntas repetidas con igual resultado)
    """
    if not registros:
        return {}

    total = len(registros)
    errores = sum(1 for r in registros if r.get("error"))
    latencias = [r["latencia_seg"] for r in registros if "latencia_seg" in r]

    # Latencia por herramienta
    herramientas = {}
    for r in registros:
        h = r.get("herramienta", "desconocida")
        herramientas.setdefault(h, []).append(r["latencia_seg"])
    lat_por_herramienta = {
        h: round(sum(v) / len(v), 3) for h, v in herramientas.items()
    }

    # Consistencia: misma pregunta → misma respuesta
    respuestas_por_pregunta: dict[str, set] = {}
    for r in registros:
        p = r.get("pregunta", "").strip().lower()
        resp = r.get("respuesta", "").strip()
        respuestas_por_pregunta.setdefault(p, set()).add(resp)

    preguntas_repetidas = {p: v for p, v in respuestas_por_pregunta.items() if len(v) > 0}
    consistentes = sum(1 for v in preguntas_repetidas.values() if len(v) == 1)
    consistencia = round(consistentes / len(preguntas_repetidas) * 100, 1) if preguntas_repetidas else 100.0

    return {
        "total_ejecuciones": total,
        "total_errores": errores,
        "tasa_error_pct": round(errores / total * 100, 1),
        "precision_proxy_pct": round((total - errores) / total * 100, 1),
        "latencia_promedio_seg": round(sum(latencias) / len(latencias), 3) if latencias else 0,
        "latencia_max_seg": round(max(latencias), 3) if latencias else 0,
        "latencia_min_seg": round(min(latencias), 3) if latencias else 0,
        "latencia_por_herramienta": lat_por_herramienta,
        "consistencia_pct": consistencia,
    }