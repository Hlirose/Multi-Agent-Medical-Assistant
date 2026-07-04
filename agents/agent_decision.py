"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails
from agents.memory_agent import PatientMemory, MemoryExtractor

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

config = Config()

memory = MemorySaver()

patient_memory = PatientMemory(config) if config.memory.enabled else None
memory_extractor = MemoryExtractor(config) if config.memory.enabled else None

thread_config = {"configurable": {"thread_id": "1"}}


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Decision model
    DECISION_MODEL = "gpt-4o"  # or whichever model you prefer
    
    # Vision model for image analysis
    VISION_MODEL = "gpt-4o"
    
    # Confidence threshold for responses
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    DECISION_SYSTEM_PROMPT = """你是一个智能医疗分诊系统，负责将用户查询路由到合适的专业智能体。你的工作是分析用户的请求，根据查询内容、是否存在图片和对话上下文，决定哪个智能体最适合处理。

    可用智能体：
    1. CONVERSATION_AGENT - 用于日常聊天、问候和非医疗问题。
    2. RAG_AGENT - 用于可以从已建立的医学文献中回答的特定医学知识问题。当前已导入的医学知识包括"脑肿瘤介绍"、"诊断和检测脑肿瘤的深度学习技术"、"从胸部X光诊断和检测新冠/COVID-19的深度学习技术"。
    3. WEB_SEARCH_PROCESSOR_AGENT - 用于关于最新医学进展、当前疫情或时效性医疗信息的问题。
    4. BRAIN_TUMOR_AGENT - 用于分析脑部MRI影像以检测和分割肿瘤。
    5. CHEST_XRAY_AGENT - 用于分析胸部X光影像以检测异常。
    6. SKIN_LESION_AGENT - 用于分析皮肤病变影像以分类为良性或恶性。

    根据以下指南做出决策：
    - 如果用户没有上传任何图片，始终路由到对话智能体。
    - 如果用户上传了医学影像，根据影像类型和用户查询决定哪个医疗视觉智能体合适。如果上传了影像但没有查询，始终根据影像类型路由到正确的医疗视觉智能体。
    - 如果用户询问最近的医学进展或当前的健康状况，使用网络搜索智能体。
    - 如果用户提出特定的医学知识问题，使用知识库智能体。
    - 对于日常对话、问候或非医疗问题，使用对话智能体。但如果上传了影像，始终先路由到医疗视觉智能体。

    你必须以JSON格式提供答案，结构如下：
    {{
    "agent": "智能体名称",
    "reasoning": "你选择此智能体的逐步推理过程",
    "confidence": 0.95
    }}
    """

    image_analyzer = ImageAnalysisAgent(config=config)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    agent_name: Optional[str]
    current_input: Optional[Union[str, Dict]]
    has_image: bool
    image_type: Optional[str]
    output: Optional[str]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool
    patient_id: Optional[str]
    patient_profile: Optional[Dict[str, Any]]
    patient_context: Optional[str]


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser

    def retrieve_patient_memory(state: AgentState) -> AgentState:
        """Retrieve long-term patient memory and inject into state."""
        if not patient_memory:
            return {**state, "patient_profile": None, "patient_context": ""}

        patient_id = state.get("patient_id")
        if not patient_id:
            return {**state, "patient_profile": None, "patient_context": ""}

        input_text = ""
        current_input = state.get("current_input", "")
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")

        context_data = patient_memory.retrieve_patient_context(
            patient_id=patient_id,
            query=input_text,
            top_k=config.memory.retrieval_top_k,
        )

        profile = context_data.get("profile")
        context_parts = []

        if profile:
            context_parts.append(f"[患者档案] {patient_memory._profile_to_text(profile)}")

        for event in context_data.get("relevant_events", []):
            evt = event["event"]
            context_parts.append(
                f"[历史记录-{evt.get('event_type', '事件')}] {evt.get('description', '')}"
            )

        patient_context = "\n".join(context_parts) if context_parts else ""

        if patient_context:
            print(f"[PatientMemory] Retrieved context for {patient_id}: {patient_context[:200]}...")

        return {
            **state,
            "patient_profile": profile,
            "patient_context": patient_context,
        }

    def extract_and_save_memory(state: AgentState) -> AgentState:
        """Extract medical information from conversation and save to long-term memory."""
        if not patient_memory or not memory_extractor:
            return state

        patient_id = state.get("patient_id")
        if not patient_id:
            return state

        messages = state.get("messages", [])
        if len(messages) < 2:
            return state

        conversation = memory_extractor.format_conversation(messages[-4:])
        extracted = memory_extractor.extract(conversation)

        if extracted is None:
            return state

        profile_updates = extracted.get("profile_updates", {})
        has_profile_update = any(v is not None and v != "" for v in profile_updates.values())
        if has_profile_update:
            patient_memory.save_patient_profile(patient_id, profile_updates)
            print(f"[PatientMemory] Updated profile for {patient_id}")

        for event in extracted.get("medical_events", []):
            patient_memory.save_medical_event(patient_id, event)
            print(f"[PatientMemory] Saved event for {patient_id}: {event.get('event_type', '')}")

        return state

    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"

        patient_ctx = state.get("patient_context", "")
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}
        
        {f'Patient long-term memory:\n{patient_ctx}\n' if patient_ctx else ''}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        patient_ctx = state.get("patient_context", "")
        patient_memory_section = f"\n        患者长期记忆:\n        {patient_ctx}\n" if patient_ctx else ""

        conversation_prompt = f"""User query: {input_text}

        Recent conversation context: {recent_context}
        {patient_memory_section}
        你是一个AI驱动的医疗对话助手。你的目标是与用户进行流畅且信息丰富的对话，处理日常聊天和医疗相关问题。你必须在确保医疗准确性和清晰度的同时自然地回应。

        ### 角色与能力
        - 在保持专业性的同时进行**日常对话**。
        - 使用经过验证的知识回答**医疗问题**。
        - 如有需要，将**复杂问题**路由到知识库（RAG）或网络搜索。
        - 在跟踪对话上下文的同时处理**追问**。
        - 将**医学影像**重定向到相应的AI分析智能体。

        ### 回应指南：
        1. **日常对话：**
        - 如果用户进行日常聊天（如问候、闲聊），以友好、热情的方式回应。
        - 保持回应**简洁且有趣**，除非需要详细回答。

        2. **医疗问题：**
        - 如果你**有把握**回答，请提供医学上准确的回答。
        - 确保回应**清晰、简洁、基于事实**。

        3. **追问与澄清：**
        - 维护对话历史以提供更好的回应。
        - 如果问题不清楚，在回答前先**追问**。

        4. **医学影像分析：**
        - **不要**尝试自己分析影像。
        - 如果用户提到分析、处理、检测、分割或分类任何影像中的疾病，请要求用户上传影像，以便在下一轮将其路由到相应的医疗视觉智能体。
        - 如果已上传影像，它会被路由到医疗计算机视觉智能体。阅读历史以了解诊断结果，如果用户询问有关诊断的问题，继续对话。
        - 处理后，**帮助用户解读结果**。

        5. **不确定性与伦理考量：**
        - 如果不确定，**绝不假设**医学事实。
        - 建议就严重的医疗问题咨询**持牌医疗专业人员**。
        - 避免提供**医疗诊断**或**处方**——仅提供一般性知识。

        ### 回应格式：
        - 保持**对话式但专业的语气**。
        - 需要时使用**项目符号或编号列表**以提高清晰度。
        - 如果引用外部来源（知识库/网络搜索），提及**信息来源**（例如，"根据梅奥诊所..."）。
        - 如果用户要求诊断，提醒他们**寻求医疗咨询**。
        - **请始终使用中文回答。**

        对话式LLM回应:"""

        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = MedicalRAG(config)
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context)
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = WebSearchProcessorAgent(config)

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        result = AgentConfig.image_analyzer.segment_brain_tumor(image_path)

        if result is None:
            response = AIMessage(content="上传的影像不够清晰，无法做出诊断，或者该影像不是医学影像。")
        elif result["has_tumor"]:
            area_pct = result["tumor_area_ratio"] * 100
            response = AIMessage(
                content=f"对上传的脑部MRI影像分析结果显示，**检测到肿瘤区域**。肿瘤占比约 **{area_pct:.1f}%**。分割结果图已生成。"
            )
        else:
            response = AIMessage(content="对上传的脑部MRI影像分析结果显示，**未检测到明显肿瘤区域**。")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,
            "agent_name": "BRAIN_TUMOR_AGENT"
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # classify chest x-ray into covid or normal
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        if predicted_class == "covid19":
            response = AIMessage(content="对上传的胸部X光影像分析结果显示，**COVID-19检测结果为阳性**。")
        elif predicted_class == "normal":
            response = AIMessage(content="对上传的胸部X光影像分析结果显示，**COVID-19检测结果为阴性**，即**正常**。")
        else:
            response = AIMessage(content="上传的影像不够清晰，无法做出诊断，或者该影像不是医学影像。")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: SKIN_LESION_AGENT")

        # classify chest x-ray into covid or normal
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

        if predicted_mask:
            response = AIMessage(content="以下是上传的皮肤病变影像的**分割**分析结果：")
        else:
            response = AIMessage(content="上传的影像不够清晰，无法做出诊断，或者该影像不是医学影像。")

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"{state['output'].content}\n\n**需要人工验证：**\n- 如果您是医疗专业人员：请验证输出结果。选择**是**或**否**。如果选择否，请提供意见。\n- 如果您是患者：直接点击'是'确认即可。"

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="之前的医学分析需要进一步审查。医疗专业人员已标记可能存在的不准确之处。")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("retrieve_patient_memory", retrieve_patient_memory)
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("extract_and_save_memory", extract_and_save_memory)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("retrieve_patient_memory")
    workflow.add_edge("retrieve_patient_memory", "analyze_input")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_edge("human_validation", "extract_and_save_memory")
    workflow.add_edge("extract_and_save_memory", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "extract_and_save_memory"
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "patient_id": None,
        "patient_profile": None,
        "patient_context": "",
    }


def process_query(query: Union[str, Dict], conversation_history: List[BaseMessage] = None, patient_id: str = None) -> str:
    """
    Process a user query through the agent decision system.
    
    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Optional list of previous messages, NOT NEEDED ANYMORE since the state saves the conversation history now
        
    Returns:
        Response from the appropriate agent
    """
    # Initialize the graph
    graph = create_agent_graph()

    # # Save Graph Flowchart
    # image_bytes = graph.get_graph().draw_mermaid_png()
    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    # cv2.imwrite("./assets/graph.png", decoded)
    # print("Graph flowchart saved in assets.")
    
    # Initialize state
    state = init_agent_state()
    
    # Add the current query
    state["current_input"] = query
    state["patient_id"] = patient_id

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."
    
    state["messages"] = [HumanMessage(content=query)]

    # result = graph.invoke(state, thread_config)
    result = graph.invoke(state, thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    if len(result["messages"]) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    return result