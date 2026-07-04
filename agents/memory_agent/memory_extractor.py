import json
import logging
from typing import Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Extracts structured medical information from conversation using LLM,
    then writes it to the long-term patient memory store.
    """

    EXTRACTION_PROMPT = """你是一个医疗信息提取助手。分析以下对话内容，从中提取关键的患者医疗信息。

对话内容：
{conversation}

请提取以下信息（如果对话中有的话），以JSON格式返回：
{{
    "has_new_info": true/false,
    "profile_updates": {{
        "name": "患者姓名（如提及）",
        "age": "年龄（如提及）",
        "gender": "性别（如提及）",
        "medical_history": "病史信息（如提及）",
        "allergies": "过敏信息（如提及）",
        "medications": "当前用药（如提及）",
        "notes": "其他重要备注"
    }},
    "medical_events": [
        {{
            "event_type": "diagnosis/consultation/test_result/medication_change/symptom",
            "description": "事件描述",
            "details": "详细信息"
        }}
    ]
}}

规则：
1. 只提取对话中明确提到的信息，不要推测
2. 如果对话中没有新的医疗信息，设置 has_new_info 为 false
3. medical_events 列表可以为空
4. profile_updates 中未提及的字段设为 null
5. 只返回JSON，不要其他文字"""

    def __init__(self, config):
        self.config = config
        self.llm = config.memory.llm

    def extract(self, conversation: str) -> Optional[Dict[str, Any]]:
        """
        Extract medical information from a conversation string.

        Args:
            conversation: The conversation text to analyze

        Returns:
            Extracted information dict, or None if extraction fails
        """
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个精确的医疗信息提取系统。只返回JSON格式数据。"),
                ("human", self.EXTRACTION_PROMPT),
            ])

            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"conversation": conversation})

            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            if not result.get("has_new_info", False):
                logger.info("No new medical information found in conversation")
                return None

            logger.info("Extracted medical information from conversation")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse extraction result as JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error during memory extraction: {e}")
            return None

    def format_conversation(self, messages: list) -> str:
        """
        Format a list of LangChain messages into a readable conversation string.

        Args:
            messages: List of HumanMessage/AIMessage objects

        Returns:
            Formatted conversation string
        """
        parts = []
        for msg in messages:
            role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
            parts.append(f"{role}: {msg.content}")
        return "\n".join(parts)