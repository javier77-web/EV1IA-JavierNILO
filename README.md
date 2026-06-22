# 📦 Asistente RAG — Starken (Proyecto IA)

Proyecto desarrollado para la asignatura **ISY0101 — Ingeniería de Soluciones con IA (DuocUC)**.

Este sistema implementa un **RAG (Retrieval-Augmented Generation)** para responder preguntas sobre envíos, tarifas y reclamos de Starken usando documentos locales.

Incluye:

* Backend (agente LangChain con herramientas y memoria)
* Interfaz web (Streamlit)
* Dashboard de monitoreo (Streamlit multipage)
* Base de conocimiento en archivos `.txt`
* Sistema de métricas y logs automáticos

---

# 🧠 ¿Cómo funciona?

```
documentos txt
      ↓
carga de documentos (loader)
      ↓
chunking (division en partes)
      ↓
embeddings (vectores)
      ↓
chroma db (base vectorial)
      ↓
agente langchain (tool-calling)
      ├── buscar_informacion
      ├── calcular_tarifa
      └── registrar_reclamo
      ↓
observability.py (métricas y logs)
      ↓
modelo gpt-4o-mini
      ↓
respuesta al usuario
```

---

# 📂 Estructura del proyecto

```
files/
│
├── app.py                  # interfaz streamlit
├── agent.py                # agente langchain con herramientas
├── tools.py                # herramientas del agente
├── memory.py               # memoria corto y largo plazo
├── observability.py        # métricas, logs y trazabilidad
├── rag_starken.py          # pipeline rag original (ev1)
│
├── pages/
│   └── dashboard.py        # dashboard de observabilidad
│
├── logs/
│   └── metricas.jsonl      # registro automático de ejecuciones
│
├── politica_envios.txt     # datos de prueba
├── tarifario_faq.txt       # datos de prueba
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Requisitos

* Python 3.13 (IMPORTANTE, no usar 3.14)
* pip
* Git
* Microsoft C++ Build Tools (requerido por ChromaDB en Windows)

---

# 🚀 Instalación paso a paso

## 1. clonar repositorio

```bash
git clone https://github.com/javier77-web/EV1IA-JavierNILO.git
cd EV1IA-JavierNILO/files
```

---

## 2. crear entorno virtual (recomendado)

### windows (powershell)

```powershell
py -3.13 -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### verificar version

```bash
python --version
```

Debe mostrar:

```
Python 3.13.x
```

---

## 3. instalar dependencias

```bash
pip install --upgrade pip
pip install streamlit
pip install langchain==0.3.25
pip install langchain-community==0.3.23
pip install langchain-openai==0.3.16
pip install langchain-core==0.3.86
pip install langchain-text-splitters
pip install chromadb==0.5.23
pip install numpy==2.1.0
pip install python-dotenv
```

---

## 4. configurar token (IMPORTANTE)

Crear archivo `.env` en la carpeta `files/`:

```
GITHUB_TOKEN=tu_token_aqui
```

👉 El token se obtiene desde GitHub:
Settings → Developer settings → Personal access tokens

---

# ▶️ Ejecución

## 🔹 opción 1: backend (consola)

```bash
py -3.13 rag_starken.py
```

Esto ejecuta pruebas automáticas del sistema RAG.

---

## 🔹 opción 2: interfaz web (recomendada)

```bash
py -3.13 -m streamlit run app.py
```

Luego abrir en navegador:

```
http://localhost:8501
```

---
 
## 🔹 opción 3: dashboard de observabilidad
 
Con la app corriendo, hacer clic en **"dashboard"** en el menú lateral de Streamlit.
 
O acceder directamente:
 
```
http://localhost:8501/dashboard
```
 
> El dashboard requiere haber hecho al menos una consulta en el chat para generar `logs/metricas.jsonl`.
---

# 💬 Ejemplos de preguntas

* que hago si mi paquete llega dañado
* cuanto demora un envio
* cuanto cuesta enviar un paquete
* puedo enviar alimentos

---

# 🛠️ Tecnologías usadas

* python 3.13
* langchain
* chromadb
* openai (via github models)
* streamlit