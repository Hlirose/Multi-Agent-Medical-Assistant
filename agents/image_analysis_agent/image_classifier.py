import os
import json
import base64
from mimetypes import guess_type

from typing import TypedDict
from langchain_core.output_parsers import JsonOutputParser

class ClassificationDecision(TypedDict):
    """Output structure for the decision agent."""
    image_type: str
    reasoning: str
    confidence: float

class ImageClassifier:
    """Uses GPT-4o Vision to analyze images and determine their type."""
    
    def __init__(self, vision_model):
        self.vision_model = vision_model
        self.json_parser = JsonOutputParser(pydantic_object=ClassificationDecision)
        
    def local_image_to_data_url(self, image_path: str) -> str:
        """
        Get the url of a local image
        """
        mime_type, _ = guess_type(image_path)

        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

        return f"data:{mime_type};base64,{base64_encoded_data}"
    
    def classify_image(self, image_path: str) -> str:
        """Analyzes the image to classify it as a medical image and determine it's type."""
        print(f"[ImageAnalyzer] Analyzing image: {image_path}")

        vision_prompt = [
            {"role": "system", "content": "你是医学影像分析专家。请分析上传的图片。"},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    """
                    判断这是否是医学影像。如果是，请将其分类为：
                    'BRAIN MRI SCAN'、'CHEST X-RAY'、'SKIN LESION' 或 'OTHER'。如果不是医学影像，返回 'NON-MEDICAL'。
                    你必须以JSON格式提供答案，结构如下：
                    {{
                    "image_type": "影像类型",
                    "reasoning": "你选择此类型的逐步推理过程",
                    "confidence": 0.95
                    }}
                    """
                )},
                {"type": "image_url", "image_url": {"url": self.local_image_to_data_url(image_path)}}
            ]}
        ]
        
        # Invoke LLM to classify the image
        response = self.vision_model.invoke(vision_prompt)

        try:
            # Ensure the response is parsed as JSON
            response_json = self.json_parser.parse(response.content)
            return response_json  # Returns a dictionary instead of a string
        except json.JSONDecodeError:
            print("[ImageAnalyzer] Warning: Response was not valid JSON.")
            return {"image_type": "unknown", "reasoning": "Invalid JSON response", "confidence": 0.0}

        # return response.content.strip().lower()