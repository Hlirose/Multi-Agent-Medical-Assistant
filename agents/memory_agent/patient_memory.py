import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class PatientMemory:
    """
    Long-term patient memory backed by Qdrant vector store.
    Stores structured patient profiles and medical history for cross-session retrieval.
    """

    COLLECTION_NAME = "patient_memory"
    EMBEDDING_DIM = 384

    def __init__(self, config):
        self.config = config
        self.memory_config = config.memory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        self._init_client()

    def _init_client(self):
        local_path = self.memory_config.local_path
        os.makedirs(local_path, exist_ok=True)
        self.client = QdrantClient(path=local_path)
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]
        if self.COLLECTION_NAME not in names:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created patient memory collection: {self.COLLECTION_NAME}")

    def _embed(self, text: str) -> List[float]:
        return self.embedding_model.embed_query(text)

    def save_patient_profile(self, patient_id: str, profile: Dict[str, Any]) -> None:
        """
        Save or update a patient profile.
        Profile fields: name, age, gender, medical_history, allergies, medications, etc.
        """
        profile_text = self._profile_to_text(profile)
        vector = self._embed(profile_text)

        existing = self._get_existing_point(patient_id)
        if existing:
            merged = self._merge_profile(existing["payload"]["profile"], profile)
            profile = merged

        point = PointStruct(
            id=patient_id,
            vector=vector,
            payload={
                "profile": profile,
                "profile_text": profile_text,
                "updated_at": datetime.now().isoformat(),
                "type": "profile",
            },
        )
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=[point])
        logger.info(f"Saved patient profile for {patient_id}")

    def save_medical_event(self, patient_id: str, event: Dict[str, Any]) -> None:
        """
        Save a medical event (diagnosis, test result, consultation summary).
        Event fields: event_type, description, date, agent, details
        """
        event_text = f"{event.get('event_type', 'event')}: {event.get('description', '')}"
        vector = self._embed(event_text)

        event_id = f"{patient_id}_event_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"

        point = PointStruct(
            id=event_id,
            vector=vector,
            payload={
                "patient_id": patient_id,
                "event": event,
                "event_text": event_text,
                "timestamp": datetime.now().isoformat(),
                "type": "event",
            },
        )
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=[point])
        logger.info(f"Saved medical event for {patient_id}: {event.get('event_type', '')}")

    def retrieve_patient_context(self, patient_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve relevant patient context for a given query.
        Returns both the patient profile and relevant medical events.
        """
        result = {
            "profile": None,
            "relevant_events": [],
        }

        profile = self.get_patient_profile(patient_id)
        if profile:
            result["profile"] = profile

        query_vector = self._embed(query)
        search_results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(key="patient_id", match=MatchValue(value=patient_id)),
                    FieldCondition(key="type", match=MatchValue(value="event")),
                ]
            ),
            limit=top_k,
        )

        for hit in search_results:
            result["relevant_events"].append({
                "event": hit.payload["event"],
                "score": hit.score,
                "timestamp": hit.payload.get("timestamp", ""),
            })

        return result

    def get_patient_profile(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get the full patient profile."""
        existing = self._get_existing_point(patient_id)
        if existing and existing["payload"].get("type") == "profile":
            return existing["payload"]["profile"]
        return None

    def _get_existing_point(self, point_id: str) -> Optional[Dict]:
        """Retrieve an existing point by ID."""
        try:
            points = self.client.retrieve(
                collection_name=self.COLLECTION_NAME,
                ids=[point_id],
            )
            if points:
                return {"id": points[0].id, "payload": points[0].payload}
        except Exception as e:
            logger.debug(f"Point {point_id} not found: {e}")
        return None

    def _profile_to_text(self, profile: Dict[str, Any]) -> str:
        """Convert structured profile to searchable text."""
        parts = []
        if profile.get("name"):
            parts.append(f"Name: {profile['name']}")
        if profile.get("age"):
            parts.append(f"Age: {profile['age']}")
        if profile.get("gender"):
            parts.append(f"Gender: {profile['gender']}")
        if profile.get("medical_history"):
            parts.append(f"Medical History: {profile['medical_history']}")
        if profile.get("allergies"):
            parts.append(f"Allergies: {profile['allergies']}")
        if profile.get("medications"):
            parts.append(f"Medications: {profile['medications']}")
        if profile.get("notes"):
            parts.append(f"Notes: {profile['notes']}")
        return " | ".join(parts) if parts else "empty profile"

    def _merge_profile(self, existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Merge new profile data into existing profile, appending list fields."""
        merged = existing.copy()
        for key, value in new.items():
            if value is None or value == "":
                continue
            if key in merged and isinstance(merged[key], str) and isinstance(value, str):
                if value.lower() not in merged[key].lower():
                    merged[key] = f"{merged[key]}; {value}"
            else:
                merged[key] = value
        return merged