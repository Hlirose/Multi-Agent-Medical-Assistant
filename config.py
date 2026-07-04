"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class. 
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ZHIPU_MODEL = "glm-4-flash"
ZHIPU_VISION_MODEL = "glm-4v-flash"

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.1
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.7
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.3
        )
        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 384
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self.llm = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.3
        )
        self.summarizer_model = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.5
        )
        self.chunker_model = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.0
        )
        self.response_generator_model = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.3
        )
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.brain_tumor_segmentation_output_path = "./uploads/brain_tumor_output/segmentation_plot.png"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_VISION_MODEL,
            temperature=0.1
        )

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")  # Replace with your actual key
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"    # Default voice ID (Rachel)

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5  # max upload size in MB

class DatabaseConfig:
    def __init__(self):
        self.db_path = "./data/medical_assistant.db"
        self.conversation_history_limit = 20

class MemoryConfig:
    def __init__(self):
        self.enabled = True
        self.local_path = "./data/patient_memory_db"
        self.llm = ChatOpenAI(
            base_url=ZHIPU_BASE_URL,
            api_key=ZHIPU_API_KEY,
            model=ZHIPU_MODEL,
            temperature=0.0
        )
        self.short_term_window = 6
        self.retrieval_top_k = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        # self.max_chat_history = 50
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.memory = MemoryConfig()
        self.database = DatabaseConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVIL_API_KEY")
        self.max_conversation_history = 20

# # Example usage
# config = Config()