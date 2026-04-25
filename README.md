# 📦 Agente RAG — Asistente Conversacional Starken

Proyecto desarrollado para la asignatura **ISY0101 — Ingeniería de Soluciones con IA** (DuocUC, 2025).

Implementa un pipeline **RAG (Retrieval-Augmented Generation)** con agente LLM para responder consultas de clientes de Starken sobre envíos, tarifas, devoluciones y reclamos.

---

## 🏗️ Arquitectura

```
Documentos (TXT/PDF)
      ↓
Document Loader (LangChain)
      ↓
Text Splitter → chunks de 500 tokens
      ↓
Embeddings (text-embedding-3-small)
      ↓
Vector Store (ChromaDB local)
      ↓
Retriever (top-3 chunks por similitud coseno)
      ↓
Prompt Builder (contexto + pregunta)
      ↓
LLM GPT-4o-mini (GitHub Models)
      ↓
Respuesta + fuente citada → Streamlit
```

---

## ⚙️ Requisitos

- Python 3.10+
- Token de GitHub con acceso a GitHub Models Marketplace

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/starken-rag.git
cd starken-rag
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar token de GitHub
```bash
# Linux/Mac
export GITHUB_TOKEN=tu_token_aqui

# Windows (PowerShell)
$env:GITHUB_TOKEN="tu_token_aqui"
```

### 5a. Ejecutar pipeline desde terminal
```bash
python src/rag_starken.py
```

### 5b. Ejecutar interfaz web (Streamlit)
```bash
streamlit run src/app.py
```
Abre tu navegador en `http://localhost:8501`

---

## 📂 Estructura del proyecto

```
starken-rag/
├── data/
│   ├── politica_envios.txt      # Política interna simulada
│   └── tarifario_faq.txt        # Tarifario y FAQ públicos
├── src/
│   ├── rag_starken.py           # Pipeline RAG principal
│   └── app.py                   # Interfaz Streamlit
├── requirements.txt
└── README.md
```

---

## 🧪 Ejemplos de consultas probadas

| Pregunta | Fuente recuperada |
|---|---|
| ¿Qué hago si mi paquete llegó dañado? | politica_envios.txt |
| ¿Cuánto cuesta enviar 3 kg? | tarifario_faq.txt |
| ¿Cuánto demora un envío a Punta Arenas? | tarifario_faq.txt |
| ¿Puedo enviar alimentos? | politica_envios.txt |

---

## 🛠️ Stack tecnológico

| Componente | Tecnología |
|---|---|
| Framework RAG | LangChain 0.3 |
| LLM | GPT-4o-mini (GitHub Models) |
| Embeddings | text-embedding-3-small |
| Vector Store | ChromaDB (local) |
| Interfaz | Streamlit |
| Lenguaje | Python 3.10+ |

---

## ⚠️ Uso de IA

Este proyecto fue desarrollado con apoyo de herramientas de IA para estructuración de código y documentación. Todas las decisiones técnicas y de diseño son propias del equipo. Referencia: https://bibliotecas.duoc.cl/ia
