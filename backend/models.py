from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class InsightRequest(BaseModel):
    questions: List[str]
    controls: Optional[Dict[str, Any]] = None

class InsightResponse(BaseModel):
    summary: str
    risks: List[str]
    actions: List[str]
    caveats: List[str]
    answers: Optional[List[str]] = None  # <— add this
