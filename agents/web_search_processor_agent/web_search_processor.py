import os
from .web_search_agent import WebSearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = WebSearchAgent(config)
        
        # Initialize LLM for processing web search results
        self.llm = config.web_search.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""以下是我们对话的最近几条消息：

        {chat_history}

        用户提出了以下问题：

        {query}

        仅当过去的对话与当前查询相关时，才将它们总结为一个完整的问题，以便用于网络搜索。
        保持简洁，确保捕捉讨论背后的关键意图。
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        # print(f"[WebSearchProcessor] Fetching web search results for: {query}")
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        # print("Web Search Query Prompt:", web_search_query_prompt)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        # print("Web Search Query:", web_search_query)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search(web_search_query.content)

        # print(f"[WebSearchProcessor] Fetched results: {web_results}")
        
        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "你是一个专门处理医疗信息的AI助手。以下是为用户查询检索到的网络搜索结果。"
            "请总结并生成一个有帮助的、简洁的回应。"
            "仅使用可靠的来源，确保医疗准确性。请使用中文回答。\n\n"
            f"查询：{query}\n\n网络搜索结果：\n{web_results}\n\n回应："
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        
        return response