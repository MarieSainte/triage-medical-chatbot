
from pydantic import BaseModel, Field
from typing import List

class TriageRequest(BaseModel):
    symptomes: str

class TriageResponse(BaseModel):
    triage: str



class TriageAnalyse(BaseModel):
    priorite: str = Field(desc="Niveau d'urgence : URGENCE, RELATIVE ou FAIBLE")
    justification: str = Field(desc="Pourquoi ce niveau d'urgence")
    recommandation: str = Field(desc="Action immédiate à faire")
    liste_des_symptomes: List[str] = Field(desc="Liste des symptômes identifiés")

class QuestionSuivi(BaseModel):
    question: str = Field(desc="La question à poser pour obtenir plus d'infos")