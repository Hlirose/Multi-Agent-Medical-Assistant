<div align="center">

<h1 align="center"><strong>Multi-Agent Medical Assistant</strong></h1>
<h6 align="center">AI-powered multi-agentic system for medical diagnosis and assistance</h6>

![Python - Version](https://img.shields.io/badge/PYTHON-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangGraph - Version](https://img.shields.io/badge/LangGraph-0.3+-teal?style=for-the-badge&logo=langgraph)
![LangChain - Version](https://img.shields.io/badge/LangChain-0.3+-teal?style=for-the-badge&logo=langchain)
![Qdrant Client - Version](https://img.shields.io/badge/Qdrant-1.13+-red?style=for-the-badge&logo=qdrant)
![Pydantic - Version](https://img.shields.io/badge/Pydantic-2.10+-red?style=for-the-badge&logo=pydantic)
![FastAPI - Version](https://img.shields.io/badge/FastAPI-0.115+-teal?style=for-the-badge&logo=fastapi)
![Docling - Version](https://img.shields.io/badge/Docling-3.1+-orange?style=for-the-badge&logo=docling)

</div>

----

## Table of Contents

- [Overview](#overview)
- [Technical Flow Chart](#technical-flowchart)
- [Key Features](#key-features)
- [Tech Stack](#technology-stack)
- [Installation and Setup](#installation-setup)
  - [Using Docker](#docker-setup)
  - [Manual Installation](#manual-setup)
- [Usage](#usage)
- [License](#license)

----

## Overview <a name="overview"></a>

The **Multi-Agent Medical Assistant** is an **AI-powered chatbot** designed to assist with **medical diagnosis, research, and patient interactions**.

Powered by Multi-Agent Intelligence, this system integrates:
- **Large Language Models (LLMs)** - GLM-4 for dialogue, RAG generation, memory extraction; GLM-4V for image type classification
- **Computer Vision Models** - U-Net for brain tumor segmentation and skin lesion segmentation; DenseNet-121 for chest X-ray COVID classification
- **Retrieval-Augmented Generation (RAG)** - Qdrant hybrid retrieval (BGE-small dense + BM25 sparse) with Cross-Encoder reranking
- **Real-time Web Search** - Tavily API for up-to-date medical insights when RAG confidence is low
- **Long/Short-Term Memory** - Qdrant-based patient profile and medical event storage with semantic retrieval
- **Input/Output Guardrails** - LLM-based safety checks for user input and model output
- **Human-in-the-Loop Validation** - Medical professionals verify AI-based medical image diagnoses
- **Voice Interaction** - ElevenLabs TTS and Whisper STT for speech-based interaction

----

## Technical Flow Chart <a name="technical-flowchart"></a>

```
User Input
    |
    v
[Input Guardrails] --UNSAFE--> Rejected
    |
    SAFE
    v
[Retrieve Patient Memory] --> Patient Context from Qdrant
    |
    v
[Analyze Input & Route]
    |
    +--> [RAG Agent] --> Confidence >= 0.4? --YES--> Response
    |         |                                          |
    |         NO (low confidence)                        |
    |         v                                          |
    |    [Web Search Agent] ---------------------------->|
    |                                                    |
    +--> [Conversation Agent] -------------------------->|
    |                                                    |
    +--> [Brain Tumor Agent] (U-Net Segmentation) ------>|
    |                                                    |
    +--> [Chest X-Ray Agent] (DenseNet-121 Classification)-->|
    |                                                    |
    +--> [Skin Lesion Agent] (U-Net Segmentation) ------>|
    |                                                    |
    v                                                    |
[Extract & Save Memory] --> Qdrant Patient Memory       |
    |                                                    |
    v                                                    |
[Output Guardrails] ------------------------------------>|
    |
    v
  Response to User
```

----

## Key Features <a name="key-features"></a>

- **Multi-Agent Architecture**: 6 specialized agents orchestrated by LangGraph state graph with confidence-based routing and agent-to-agent handoff

- **Advanced RAG Pipeline**:
  - Docling-based PDF parsing extracting text, tables, and images
  - LLM-based image summarization and markdown formatting
  - LLM-based semantic chunking with structural boundary awareness
  - LLM-based query expansion with related medical domain terms
  - Qdrant hybrid search combining BM25 sparse keyword search with BGE-small dense embedding vector search
  - Cross-Encoder (BGE-reranker-base) reranking of retrieved document chunks
  - Confidence-based agent-to-agent handoff between RAG and Web Search
  - Source document links and reference images provided with responses

- **Medical Imaging Analysis**:
  - Brain Tumor Detection (U-Net Semantic Segmentation)
  - Chest X-ray COVID Classification (DenseNet-121)
  - Skin Lesion Segmentation (U-Net)

- **Long/Short-Term Memory**:
  - Short-term: LangGraph MessagesState for current session context
  - Long-term: Qdrant vector database for patient profiles and medical events with semantic retrieval across sessions

- **Input & Output Guardrails**: LLM-based safety checks filtering 47+ categories of unsafe input and 10 categories of unsafe output

- **Real-time Research Integration**: Web search agent retrieves the latest medical research when RAG confidence is low

- **Voice Interaction**: Speech-to-text and text-to-speech powered by ElevenLabs API

- **Human-in-the-Loop Verification**: Medical professionals validate AI-based image diagnoses before final output

- **Database Persistence**: SQLite with SQLAlchemy ORM for conversation history and medical analysis records

----

## Technology Stack <a name="technology-stack"></a>

| Component | Technologies |
|-----------|-------------|
| **Backend Framework** | FastAPI, Uvicorn |
| **Agent Orchestration** | LangGraph, LangChain |
| **LLM** | GLM-4 (ZhipuAI), GLM-4V (multimodal) |
| **Document Parsing** | Docling |
| **Embedding Model** | BGE-small-zh (dense), FastEmbed BM25 (sparse) |
| **Reranker** | BGE-reranker-base (Cross-Encoder) |
| **Vector Database** | Qdrant (hybrid retrieval) |
| **Relational Database** | SQLite, SQLAlchemy |
| **Medical Imaging** | PyTorch |
| | Brain Tumor: U-Net Semantic Segmentation |
| | Chest X-ray: DenseNet-121 Image Classification |
| | Skin Lesion: U-Net Semantic Segmentation |
| **Image Processing** | OpenCV, Pillow |
| **Web Search** | Tavily API |
| **Speech Processing** | ElevenLabs API (TTS), Whisper (STT) |
| **Guardrails** | LangChain + LLM-based |
| **Frontend** | HTML, CSS, JavaScript, Bootstrap 5.3, marked.js |
| **Deployment** | Docker |

----

## Installation & Setup <a name="installation-setup"></a>

### Option 1: Using Docker <a name="docker-setup"></a>

#### Prerequisites:

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- API keys for the required services

#### 1. Clone the Repository

```bash
git clone https://github.com/Hlirose/Multi-Agent-Medical-Assistant.git
cd Multi-Agent-Medical-Assistant
```

#### 2. Create Environment File

Create a `.env` file in the root directory and add the following API keys:

> [!WARNING]
> Ensure the API keys in the `.env` file are correct and have the necessary permissions.
> No trailing whitespaces after variable names.

```bash
# LLM Configuration (ZhipuAI GLM-4)
ZHIPUAI_API_KEY=

# Embedding Model Configuration (BGE-small-zh)
# Uses ZhipuAI embedding API by default

# Speech API Key (Free credits available with new ElevenLabs Account)
ELEVEN_LABS_API_KEY=

# Web Search API Key (Free credits available with new Tavily Account)
TAVILY_API_KEY=

# Hugging Face Token - for reranker model BGE-reranker-base
HUGGINGFACE_TOKEN=

# (OPTIONAL) If using Qdrant server version, local does not require API key
QDRANT_URL=
QDRANT_API_KEY=
```

#### 3. Build the Docker Image

```bash
docker build -t medical-assistant .
```

#### 4. Run the Docker Container

```bash
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env medical-assistant
```

The application will be available at: [http://localhost:8000](http://localhost:8000)

#### 5. Ingest Data into Vector DB from Docker Container

- To ingest a single document:
```bash
docker exec medical-assistant-app python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```

- To ingest multiple documents from a directory:
```bash
docker exec medical-assistant-app python ingest_rag_data.py --dir ./data/raw
```

#### Managing the Container:

Stop the container:
```bash
docker stop medical-assistant-app
```

Start the container:
```bash
docker start medical-assistant-app
```

View logs:
```bash
docker logs medical-assistant-app
```

Remove the container:
```bash
docker rm medical-assistant-app
```

#### Troubleshooting:

Check container health status:
```bash
docker inspect --format='{{.State.Health.Status}}' medical-assistant-app
```

If the container fails to start, check the logs:
```bash
docker logs medical-assistant-app
```

----

### Option 2: Without Using Docker <a name="manual-setup"></a>

#### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Multi-Agent-Medical-Assistant.git
cd Multi-Agent-Medical-Assistant
```

#### 2. Create & Activate Virtual Environment

- If using conda:
```bash
conda create --name medical-assistant python=3.11
conda activate medical-assistant
```

- If using python venv:
```bash
python -m venv medical-assistant
source medical-assistant/bin/activate  # For Mac/Linux
medical-assistant\Scripts\activate     # For Windows
```

#### 3. Install Dependencies

> [!IMPORTANT]
> ffmpeg is required for speech service to work.

- If using conda:
```bash
conda install -c conda-forge ffmpeg
pip install -r requirements.txt
```

- If using python venv:
```bash
winget install ffmpeg
pip install -r requirements.txt
```

#### 4. Set Up API Keys

Create a `.env` file and add the required API keys as shown in Option 1.

#### 5. Run the Application

```bash
python app.py
```

The application will be available at: [http://localhost:8000](http://localhost:8000)

#### 6. Ingest Data into the Vector DB

- To ingest one document at a time:
```bash
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```

- To ingest multiple documents from a directory:
```bash
python ingest_rag_data.py --dir ./data/raw
```

----

## Usage <a name="usage"></a>

> [!NOTE]
> 1. The first run can be jittery and may get errors - be patient and check the console for ongoing downloads and installations.
> 2. On the first run, many models will be downloaded - computer vision models, cross-encoder reranker model, embedding models, etc.
> 3. Once they are completed, retry. Everything should work seamlessly.

- Upload medical images for **AI-based diagnosis**. Upload images from `sample_images` folder to try out.
- Ask medical queries to leverage **retrieval-augmented generation (RAG)** or **web search** for latest information.
- Use **voice-based** interaction (speech-to-text and text-to-speech).
- Review AI-generated insights with **human-in-the-loop verification**.

----

## License <a name="license"></a>

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.

Based on [Multi-Agent-Medical-Assistant](https://github.com/souvikmajumder26/Multi-Agent-Medical-Assistant) by Souvik Majumder.

----

<p align="right">
 <a href="#top"><b>Return to top</b></a>
</p>
