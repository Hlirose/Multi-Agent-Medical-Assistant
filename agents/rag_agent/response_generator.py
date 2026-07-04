import logging
from typing import List, Dict, Any, Optional, Union

class ResponseGenerator:
    """
    Generates responses based on retrieved context and user query.
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            config: Configuration object
            llm: Large language model for response generation
        """
        self.logger = logging.getLogger(__name__)
        self.response_generator_model = config.rag.response_generator_model
        self.include_sources = getattr(config.rag, "include_sources", True)

    def _build_prompt(
            self,
            query: str, 
            context: str,
            chat_history: Optional[List[Dict[str, str]]] = None
        ) -> str:
        """
        Build the prompt for the language model.
        
        Args:
            query: User query
            context: Formatted context from retrieved documents
            chat_history: Optional chat history
            
        Returns:
            Complete prompt string
        """

        table_instructions = """
        部分检索到的信息以表格格式呈现。使用表格中的信息时：
        1. 使用正确的Markdown表格格式呈现表格数据，包含表头，如下所示：
            | 列1 | 列2 | 列3 |
            |------|------|------|
            | 值1  | 值2  | 值3  |
        2. 重新格式化表格结构，使其更易于阅读和理解
        3. 如果在重新格式化表格时引入了新组件，请明确说明
        4. 在回应中清楚地解读表格数据
        5. 在呈现特定数据点时引用相关表格
        6. 如果合适，总结表格中显示的趋势或模式
        7. 如果只提到了参考编号，并且你可以从上下文中获取对应的值（如论文标题或作者），请用实际值替换参考编号
        """

        response_format_instructions = """说明：
        1. 仅根据上下文中提供的信息回答查询。
        2. 如果上下文不包含回答查询的相关信息，请声明："根据提供的上下文，我没有足够的信息来回答这个问题。"
        3. 不要使用上下文中未包含的先验知识。
        4. 简洁准确。
        5. 根据检索到的知识，提供带有标题、子标题和表格结构（如需要）的Markdown格式结构良好的回应。保持标题和子标题小尺寸。
        6. 仅提供对聊天机器人回复有意义的部分。例如，不要明确提及参考文献。
        7. 如果涉及数值，确保回应中包含上下文中出现的精确值。不要编造数值。
        8. 不要在答案或回应中重复问题。
        9. 请始终使用中文回答。"""
            
        prompt = f"""你是一个医疗助手，基于经过验证的医学来源提供准确的信息。

        以下是我们对话的最近几条消息：
        
        {chat_history}

        用户提出了以下问题：
        {query}

        我检索了以下信息来帮助回答这个问题：

        {context}

        {table_instructions}

        {response_format_instructions}

        基于提供的信息，请全面但简洁地回答用户的问题。
        如果信息不包含答案，请承认可用信息的局限性。

        不要提供上下文中不存在的来源链接。不要编造任何来源链接。

        医疗助手回应:"""

        return prompt

    def generate_response(
            self,
            query: str,
            retrieved_docs: List[Dict[str, Any]],
            picture_paths: List[str],
            chat_history: Optional[List[Dict[str, str]]] = None,
        ) -> Dict[str, Any]:
        """
        Generate a response based on retrieved documents.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved document dictionaries
            chat_history: Optional chat history
            
        Returns:
            Dict containing response text and source information
        """
        try:
           
            # Extract content from documents for context
            doc_texts = [doc["content"] for doc in retrieved_docs]
            
            # Combine retrieved documents into a single context
            context = "\n\n===DOCUMENT SECTION===\n\n".join(doc_texts)
            
            # Build the prompt
            prompt = self._build_prompt(query, context, chat_history)
            
            # Generate response
            response = self.response_generator_model.invoke(prompt)
            
            # Extract sources for citation
            sources = self._extract_sources(retrieved_docs) if hasattr(self, 'include_sources') and self.include_sources else []
            
            # Calculate confidence
            confidence = self._calculate_confidence(retrieved_docs)

            # Add sources to response
            if hasattr(self, 'include_sources') and self.include_sources:
                response_with_source = response.content + "\n\n##### Source documents:"
                for current_source in sources:
                    source_path = current_source['path']
                    source_title = current_source['title']
                    response_with_source += f"\n- [{source_title}]({source_path})"
            else:
                response_with_source = response.content
            
            # Add picture paths to response
            response_with_source_and_picture_paths = response_with_source + "\n\n##### Reference images:"
            for picture_path in picture_paths:
                response_with_source_and_picture_paths += f"\n- [{picture_path.split('/')[-1]}]({picture_path})"
            
            # Format final response
            result = {
                "response": response_with_source_and_picture_paths,
                "sources": sources,
                "confidence": confidence
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": "抱歉，生成回应时出现错误。请尝试换一种方式提问。",
                "sources": [],
                "confidence": 0.0
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract source information from retrieved documents for citation.
        
        Args:
            documents: List of retrieved document dictionaries
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        seen_sources = set()  # Track unique sources to avoid duplicates
        
        for doc in documents:
            # Extract source and source_path
            source = doc.get("source")
            source_path = doc.get("source_path")
            
            # Skip if no source information is available
            if not source:
                continue
                
            # Create a unique identifier for this source
            source_id = f"{source}|{source_path}"
            
            # Skip if we've already included this source
            if source_id in seen_sources:
                continue
                
            # Add to our sources list
            source_info = {
                "title": source,
                "path": source_path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", 0.0)))
            }
            
            sources.append(source_info)
            seen_sources.add(source_id)
        
        # Sort sources by score from highest to lowest
        sources.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Format the final sources list, removing the scores which were just used for sorting
        formatted_sources = []
        for source in sources:
            formatted_source = {
                "title": source["title"],
                "path": source["path"]
            }
            formatted_sources.append(formatted_source)
            
        return formatted_sources

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        
        Args:
            documents: Retrieved documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not documents:
            return 0.0
            
        # Use combined score (both reranker and cosine similarity) if available, otherwise use original score
        if "combined_score" in documents[0]:
            scores = [doc.get("combined_score", 0) for doc in documents[:3]]
        elif "rerank_score" in documents[0]:
            scores = [doc.get("rerank_score", 0) for doc in documents[:3]]
        else:
            scores = [doc.get("score", 0) for doc in documents[:3]]
            
        # Average of top 3 document scores or fewer if less than 3
        return sum(scores) / len(scores) if scores else 0.0