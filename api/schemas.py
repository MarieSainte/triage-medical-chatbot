from pydantic import BaseModel

class TriageRequest(BaseModel):
    symptomes: str

class TriageResponse(BaseModel):
    triage: str