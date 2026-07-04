import logging
from typing import List, Dict, Any

class QueryExpander:
    """
    Expands user queries with medical terminology to improve retrieval.
    """
    def __init__(self, config):
        self.logger = logging.getLogger(f"{self.__module__}")
        self.config = config
        self.model = config.rag.llm
        
    def expand_query(self, original_query: str) -> Dict[str, Any]:
        """
        Expand the original query with relevant medical terms.
        
        Args:
            original_query: The user's original query
            
        Returns:
            Dictionary with original and expanded queries
        """
        self.logger.info(f"Expanding query: {original_query}")
        
        # Generate expansions - implement one of the strategies below
        expanded_query = self._generate_expansions(original_query)
        
        return {
            "original_query": original_query,
            "expanded_query": expanded_query.content
        }
    
    def _generate_expansions(self, query: str) -> str:
        """Use LLM to expand query with medical terminology."""
        prompt = f"""
        作为医学专家，请用相关的医学术语、同义词和相关概念扩展以下查询，以帮助检索相关的医学信息：
        
        用户查询：{query}
        
        仅在你认为需要时才扩展查询，否则保持用户查询不变。
        仅针对用户查询中提到的医疗或其他领域进行扩展，不要添加其他医疗领域。
        如果用户查询要求以表格格式回答，请在扩展查询中包含该要求，不要自己以表格格式回答。
        仅提供扩展后的查询，不要解释。
        """
        expansion = self.model.invoke(prompt)
        
        return expansion