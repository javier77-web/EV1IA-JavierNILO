# 📦 Agente Funcional Starken — EV2

Proyecto desarrollado para la asignatura **ISY0101 — Ingeniería de Soluciones con IA (DuocUC)**.

Este sistema implementa un **Agente Funcional basado en LangChain AgentExecutor** para responder consultas relacionadas con envíos, tarifas y reclamos de Starken.

Incluye:

* Backend del agente (AgentExecutor)
* Sistema RAG con ChromaDB
* Memoria de corto plazo
* Memoria persistente de largo plazo
* Interfaz web con Streamlit
* Base de conocimiento en archivos `.txt`

---

# 🧠 ¿Cómo funciona?

```text
Usuario (Streamlit)
        │
        ▼
   AgentExecutor
        │
        ├── Memoria corto plazo
        │
        ├── Memoria largo plazo
        │
        └── Herramientas
              │
              ├── buscar_informacion
              ├── calcular_tarifa
              └── registrar_reclamo
                     │
                     ▼
                  ChromaDB
                     │
                     ▼
                GPT-4o-mini
                     │
                     ▼
                 Respuesta
```

---

# 🔧 Herramientas disponibles

| Herramienta        | Función                                                             |
| ------------------ | ------------------------------------------------------------------- |
| buscar_informacion | Busca información relevante en la base de conocimiento mediante RAG |
| calcular_tarifa    | Calcula costos de envío según peso y destino                        |
| registrar_reclamo  | Clasifica reclamos y entrega instrucciones al usuario               |

---

# 🧠 Sistema de memoria

### Memoria de corto plazo

Permite mantener contexto durante la conversación actual utilizando el historial reciente de mensajes.

### Memoria de largo plazo

Almacena resúmenes de conversaciones en ChromaDB para recuperar información relevante entre distintas sesiones.

---

# 📂 Estructura del proyecto

```text
files/
│
├── app.py
├── agent.py
├── tools.py
├── memory.py
├── rag_starken.py
├── politica_envios.txt
├── tarifario_faq.txt
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

# ⚙️ Requisitos

* Python 3.13
* pip
* Git
* Microsoft C++ Build Tools (Windows)

> Se recomienda utilizar Python 3.13. Algunas dependencias aún presentan incompatibilidades con Python 3.14.

---

# ⚠️ Requisito adicional para Windows

ChromaDB requiere compilar ciertos componentes en C++.

Antes de instalar dependencias:

1. Descargar Visual C++ Build Tools:

https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. Ejecutar el instalador.

3. Seleccionar únicamente:

```text
Desarrollo para escritorio con C++
```

4. Completar la instalación.

5. Reiniciar PowerShell.

---

# 🚀 Instalación paso a paso

## 1. Clonar repositorio

```bash
git clone https://github.com/javier77-web/EV1IA-JavierNILO.git
```

---

## 2. Crear entorno virtual

### Windows (PowerShell)

```powershell
py -3.13 -m venv venv

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

.\venv\Scripts\Activate.ps1
```

### Verificar versión

```bash
python --version
```

Debe mostrar:

```text
Python 3.13.x
```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si aparece un error relacionado con `chroma-hnswlib`, verificar que Microsoft C++ Build Tools se encuentre correctamente instalado.

---

## 4. Configurar token (IMPORTANTE)

Crear un archivo `.env` en la carpeta principal del proyecto:

```env
GITHUB_TOKEN=tu_token_aqui
```

### Obtener token desde GitHub

1. Ingresar a GitHub.
2. Abrir Settings.
3. Seleccionar Developer Settings.
4. Ingresar a Personal Access Tokens.
5. Crear un nuevo token.
6. Copiar el token generado.
7. Guardarlo dentro del archivo `.env`.

---

# ▶️ Ejecución

## 🔹 Opción 1: Pruebas del agente desde consola

```bash
python agent.py
```

Permite ejecutar casos de prueba automáticos para validar:

* Uso de herramientas
* Recuperación de información mediante RAG
* Cálculo de tarifas
* Registro de reclamos
* Funcionamiento de la memoria

---

## 🔹 Opción 2: Interfaz web

```bash
streamlit run app.py
```

Luego abrir en el navegador:

```text
http://localhost:8501
```

---

# 💬 Ejemplos de consultas

```text
¿Cuánto cuesta enviar un paquete de 3 kg a Punta Arenas?

¿Qué debo hacer si mi paquete llegó dañado?

Mi envío lleva más de 10 días sin actualizar estado.

¿Existe servicio express para Santiago?

¿Cuánto demora un envío a regiones?

Quiero enviar un sobre de 400 gramos a Arica.
```

---

# 🛠️ Tecnologías utilizadas

* Python 3.13
* LangChain
* AgentExecutor
* ChromaDB
* GPT-4o-mini
* GitHub Models
* Streamlit
* Python-dotenv

---

# 📌 Características principales

* Selección automática de herramientas.
* Arquitectura basada en agentes.
* Recuperación de información mediante RAG.
* Memoria conversacional de corto plazo.
* Memoria persistente entre sesiones.
* Base vectorial utilizando ChromaDB.
* Interfaz web para interacción con usuarios.

