
from pydantic import BaseModel, Field
from typing import List
from datetime import datetime

class LogResponse(BaseModel):
    id: int
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True

class TriageRequest(BaseModel):
    symptomes: str

from typing import Optional, Any

class TriageResponse(BaseModel):
    status: str
    data: Optional[Any] = None
    question: Optional[str] = None
    latency: Optional[float] = None



class TriageAnalyse(BaseModel):
    priorite: str = Field(desc="Niveau d'urgence : URGENCE, RELATIVE ou FAIBLE")
    justification: str = Field(desc="Pourquoi ce niveau d'urgence")
    recommandation: str = Field(desc="Action immédiate à faire")
    liste_des_symptomes: List[str] = Field(desc="Liste des symptômes identifiés")

class QuestionSuivi(BaseModel):
    question: str = Field(desc="La question à poser pour obtenir plus d'infos")