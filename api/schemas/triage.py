
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime


class LogResponse(BaseModel):
    id: int
    question: str
    answer: str
    created_at: datetime

    class Config:
        from_attributes = True


class ChatMLMessage(BaseModel):
    """Un message au format ChatML (role + content)."""
    role: str   # "user" | "assistant"
    content: str


class TriageRequest(BaseModel):
    """
    Accepte soit un historique complet de messages ChatML,
    soit un texte brut (rétrocompatibilité).
    """
    messages: Optional[List[ChatMLMessage]] = None
    symptomes: Optional[str] = None


class TriageResponse(BaseModel):
    status: str
    data: Optional[Any] = None
    question: Optional[str] = None
    message: Optional[str] = None
    latency: Optional[float] = None


class TriageAnalyse(BaseModel):
    priorite: str = Field(description="Niveau d'urgence : URGENCE, RELATIVE ou FAIBLE")
    justification: str = Field(description="Pourquoi ce niveau d'urgence")
    recommandation: str = Field(description="Action immédiate à faire")
    liste_des_symptomes: List[str] = Field(description="Liste des symptômes identifiés")


class QuestionSuivi(BaseModel):
    question: str = Field(description="La question à poser pour obtenir plus d'infos")