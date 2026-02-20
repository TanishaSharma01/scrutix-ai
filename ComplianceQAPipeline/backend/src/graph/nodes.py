import json
import os
import logging
import re
from typing import Dict, Any, List

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# import state schema
from backend.src.graph.state import VideoAuditState, ComplianceIssue

# import service
from backend.src.services.video_indexer import VideoIndexerService

# configure the logger 
logger = logging.getLogger("scrutix-ai")
logging.basicConfig(level=logging.INFO)

# Node 1: Indexer
def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    Downloads the youtube video from the url
    Uploads to the Azure Video Indexer
    extracts the insights 
    '''

    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")

    logger.info(f"---[Node: Indexer] Processing: {video_url}---")

    local_filename = "temp_audit_video.mp4"

    try:
        vi_service = VideoIndexerService()

        # download
        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please provide a valid YouTube URL for this test.")
        
        # upload
        azure_video_id = vi_service.upload_video(local_path, video_name = video_id_input)
        logger.info(f"Upload Success. Azure Video ID: {azure_video_id}")

        # cleanup
        if os.path.exists(local_path):
            os.remove(local_path)
        
        # wait
        raw_insights = vi_service.wait_for_processing(azure_video_id)

        # extract
        clean_data = vi_service.extract_data(raw_insights)
        logger.info("---[Node: Indexer] Extraction Success. ---")

        return clean_data
    
    except Exception as e:
        logger.error(f"Video Indexing Failed: {str(e)}")
        return{
            "errors": [str(e)]
            "final_status": "FAIL",
            "transcript": "",
            "ocr_text": []
        }

# Node 2: Compliance Auditor
def audio_content_node(state: VideoAuditState) -> Dict[str, Any]:
    '''
    Performs Retrieval Augmented Generation to audit the content of the video
    '''

    logger.info("---[Node: Auditor] querying Knowledge Base & LLM---")
    transcript = state.get("transcript", "")

    if not transcript:
        logger.warning("No transcript available. Skipping audit....")
        return {
            "final_status": "FAIL",
            "final_report": "Audit skipped because video processing failed (No Transcript)."
        }
    
    # intialize clients
    llm = AzureChatOpenAI(
        azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0.0
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment = "text-embedding-3-small",
        open_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        temperature=0.0
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment = "text-embedding-3-small",
        open_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    )

    vectorstore = AzureSearch(
        azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key = os.getenv("AZURE_SEARCH_KEY"),
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"), 
        embedding_function = embeddings.embed_query
    )

    # RAG Retrieval
    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"  
    docs = vector_store.similarity_search(query_text, k=3)
    retrieved_rules = "\n\n", join([doc.page_content for doc in docs])
    





