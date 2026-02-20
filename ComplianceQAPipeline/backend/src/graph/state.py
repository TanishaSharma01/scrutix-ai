import operator
from typing import Annotated, List, Dict, Optional, Any, TypedDict

# define the schema for a single complicance result
# Error Report
class ComplianceIssue(TypedDict):
    category: str # audio, video, text
    description: str # specific detail of violation
    severity: str # CRITICAL | WARNING
    timestamp: Optional[str] 

# define the global graph state
# this defines the state that gets passed around in the agentic workflow
class VideoAuditState:
    '''
    Defines the data schema for langgraph execution content
    Main container: holds all th information about the audit
    right from the initial URL to the final report
    '''
    video_url: str
    video_id: str

    # ingestion and extraction data
    local_file_path: Optional[str]
    video_metadata: Dict[str, Any] # {"duration": 15, "resolution": "1080p"}
    transcript: Optional[str] # Fully extracted speech-to-text
    ocr_text: List[str] # text that appears at a screen cap

    # analysis output
    # stores the list of all violations found by AI
    compliance_results: Annotated[List[ComplianceIssue], operator.add]

    # final deliverables
    final_status: str # PASS | FAIL
    final_report: str # markdown format

    # system observability
    # errors: API timeout, system level errors
    # list of system level  
    errors: Annotated[List[str], operator.add]
