<div align="center">

<h1 align="center"><strong>Agent Details</strong></h1>
<h6 align="center">All implemented agents have been detailed below</h6>

</div>

---

## Table of Contents

- [Agent Architecture Overview](#agent-architecture)
- [RAG Agent](#rag-agent)
- [Web Search Agent](#web-search-agent)
- [Conversation Agent](#conversation-agent)
- [Brain Tumor Agent](#brain-tumor-agent)
- [Chest X-Ray Agent](#chest-xray-agent)
- [Skin Lesion Agent](#skin-lesion-agent)
- [Memory Agent](#memory-agent)
- [Guardrails](#guardrails)
- [Human-in-the-Loop Validation](#human-in-the-loop)
- [Research Papers and Documents Used for RAG (Citations)](#citations)

---

## Agent Architecture Overview <a name="agent-architecture"></a>

All agents are orchestrated by a LangGraph state graph defined in `agent_decision.py`. The routing logic uses LLM-based input analysis to direct queries to the appropriate agent, with confidence-based handoff between RAG and Web Search agents.

```
analyze_input --> route_to_agent
    |
    +--> RAG Agent (confidence >= 0.4) --> Response
    |         |
    |         +--> (confidence < 0.4) --> Web Search Agent --> Response
    |
    +--> Conversation Agent --> Response
    +--> Brain Tumor Agent --> Response
    +--> Chest X-Ray Agent --> Response
    +--> Skin Lesion Agent --> Response
```

---

## RAG Agent <a name="rag-agent"></a>

5-step online retrieval pipeline:
1. **Query Expansion**: LLM expands user query with related medical domain terms
2. **Hybrid Retrieval**: Qdrant combines BGE-small dense vectors + BM25 sparse vectors
3. **Cross-Encoder Reranking**: BGE-reranker-base reranks retrieved chunks by relevance
4. **LLM Generation**: GLM-4 generates answer using reranked chunks as context
5. **Confidence Calculation**: Log probability-based confidence score determines if handoff to Web Search is needed

---

## Web Search Agent <a name="web-search-agent"></a>

Activated when RAG confidence < 0.4. Uses Tavily API to search the web for up-to-date medical information, then GLM-4 summarizes the search results into a coherent response.

---

## Conversation Agent <a name="conversation-agent"></a>

Handles general medical dialogue, greetings, and non-specific queries using GLM-4 with conversation history context.

---

## Brain Tumor Agent <a name="brain-tumor-agent"></a>

U-Net semantic segmentation model for brain tumor detection from MRI scans. Outputs:
- Original MRI image
- Tumor segmentation heatmap
- Overlay image (original + segmentation mask)
- Tumor area ratio and classification

---

## Chest X-Ray Agent <a name="chest-xray-agent"></a>

DenseNet-121 image classification model for COVID-19 detection from chest X-ray images. Binary classification: covid19 / normal.

---

## Skin Lesion Agent <a name="skin-lesion-agent"></a>

U-Net semantic segmentation model for skin lesion boundary detection. Outputs overlay image with segmentation mask.

---

## Memory Agent <a name="memory-agent"></a>

Two components:
- **MemoryExtractor**: LLM-based extraction of structured medical information from conversations (patient profile updates + medical events)
- **PatientMemory**: Qdrant-based storage and semantic retrieval of patient profiles and medical events across sessions

Short-term memory: LangGraph MessagesState (current session only)
Long-term memory: Qdrant vector database (persistent across sessions)

---

## Guardrails <a name="guardrails"></a>

Dual-layer safety system:
- **Input Guardrails**: LLM-based check filtering 47+ categories of unsafe input (code injection, non-medical content, literature citation requests, etc.)
- **Output Guardrails**: LLM-based check filtering 10 categories of unsafe output (missing disclaimers, harmful medical advice, prompt leakage, etc.)

---

## Human-in-the-Loop Validation <a name="human-in-the-loop"></a>

For medical image analysis results, the system implements human validation:

Backend (`agent_decision.py`):
1. Interrupt the workflow when human validation is needed
2. Store the interrupted state in memory
3. Add endpoints to expose pending validations and submit validation decisions
4. Resume the workflow after the human has provided feedback

Frontend:
1. Check if a response needs validation (needs_validation flag)
2. If so, show a validation interface to the human reviewer
3. Send the validation decision back through the /validate endpoint
4. Continue the conversation

---

## Research Papers and Documents Used for RAG (Citations) <a name="citations"></a>

1. Saeedi, S., Rezayi, S., Keshavarz, H. et al. MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques. BMC Med Inform Decis Mak 23, 16 (2023). [https://doi.org/10.1186/s12911-023-02114-6](https://doi.org/10.1186/s12911-023-02114-6)

2. Babu Vimala, B., Srinivasan, S., Mathivanan, S.K. et al. Detection and classification of brain tumor using hybrid deep learning models. Sci Rep 13, 23029 (2023). [https://doi.org/10.1038/s41598-023-50505-6](https://doi.org/10.1038/s41598-023-50505-6)

3. Khaliki, M.Z., Basarslan, M.S. Brain tumor detection from images and comparison with transfer learning methods and 3-layer CNN. Sci Rep 14, 2664 (2024). [https://doi.org/10.1038/s41598-024-52823-9](https://doi.org/10.1038/s41598-024-52823-9)

4. Brain Tumors: an Introduction basic level, Mayfield Clinic, UCNI

5. Cleverley J, Piper J, Jones M M. The role of chest radiography in confirming covid-19 pneumonia BMJ 2020; 370 :m2426 [https://doi.org/10.1136/bmj.m2426](https://doi.org/10.1136/bmj.m2426)

6. Yasin, R., Gouda, W. Chest X-ray findings monitoring COVID-19 disease course and severity. Egypt J Radiol Nucl Med 51, 193 (2020). [https://doi.org/10.1186/s43055-020-00296-x](https://doi.org/10.1186/s43055-020-00296-x)

7. Cozzi, D., Albanesi, M., Cavigli, E. et al. Chest X-ray in new Coronavirus Disease 2019 (COVID-19) infection: findings and correlation with clinical outcome. Radiol med 125, 730-737 (2020). [https://doi.org/10.1007/s11547-020-01232-9](https://doi.org/10.1007/s11547-020-01232-9)

8. Jain, R., Gupta, M., Taneja, S. et al. Deep learning based detection and analysis of COVID-19 on chest X-ray images. Appl Intell 51, 1690-1700 (2021). [https://doi.org/10.1007/s10489-020-01902-1](https://doi.org/10.1007/s10489-020-01902-1)

9. El Houby, E.M.F. COVID-19 detection from chest X-ray images using transfer learning. Sci Rep 14, 11639 (2024). [https://doi.org/10.1038/s41598-024-61693-0](https://doi.org/10.1038/s41598-024-61693-0)

10. [Diabetes mellitus](https://www.researchgate.net/publication/270283336_Diabetes_mellitus)

11. Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC). Noel C. F. Codella, David Gutman, M. Emre Celebi, Brian Helba, Michael A. Marchetti, Stephen W. Dusza, Aadi Kalloo, Konstantinos Liopyris, Nabin Mishra, Harald Kittler, Allan Halpern. [https://doi.org/10.48550/arXiv.1710.05006](https://doi.org/10.48550/arXiv.1710.05006)

12. Zahra Mirikharaji, Kumar Abhishek, Alceu Bissoto, Catarina Barata, Sandra Avila, Eduardo Valle, M. Emre Celebi, Ghassan Hamarneh. A survey on deep learning for skin lesion segmentation. Medical Image Analysis, Volume 88, 2023, 102863, ISSN 1361-8415. [https://doi.org/10.1016/j.media.2023.102863](https://doi.org/10.1016/j.media.2023.102863)

---
