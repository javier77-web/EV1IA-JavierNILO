# 📦 Asistente RAG — Starken (Proyecto IA)

Proyecto desarrollado para la asignatura **ISY0101 — Ingeniería de Soluciones con IA (DuocUC)**.

Este sistema implementa un **RAG (Retrieval-Augmented Generation)** para responder preguntas sobre envíos, tarifas y reclamos de Starken usando documentos locales.

Incluye:

* Backend (pipeline RAG en Python)
* Interfaz web (Streamlit)
* Base de conocimiento en archivos `.txt`

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
rag manual (busqueda semantica)
      ↓
prompt + pregunta
      ↓
modelo gpt-4o-mini
      ↓
respuesta con fuentes
```

---

# 📂 Estructura del proyecto

```
files/
│
├── app.py                # interfaz streamlit
├── rag_starken.py        # backend rag manual
├── politica_envios.txt   # datos de prueba
├── tarifario_faq.txt     # datos de prueba
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Requisitos

* Python 3.13 (IMPORTANTE, no usar 3.14)
* pip
* Git

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
pip install langchain
pip install langchain-community
pip install langchain-openai
pip install langchain-text-splitters
pip install chromadb
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
python rag_starken.py
```

Esto ejecuta pruebas automáticas del sistema RAG.

---

## 🔹 opción 2: interfaz web (recomendada)

```bash
streamlit run app.py
```

Luego abrir en navegador:

```
http://localhost:8501
```

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



