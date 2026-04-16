from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from services import chatbot as service
from schemas import triage 
from database.database import get_db

router = APIRouter(prefix="/triage", tags=["Triage"])

@router.post("/ask", response_model=triage.TriageResponse)
def ask_triage(request: triage.TriageRequest, db: Session = Depends(get_db)):
    try:
        response_dict = service.generate_triage(request.symptomes)
        import json
        service.log_triage(db=db, question=request.symptomes, answer=json.dumps(response_dict, ensure_ascii=False))
        return response_dict
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne lors du triage.")

@router.get("/logs", response_model=list[triage.LogResponse])
def get_triage_logs(db: Session = Depends(get_db)):
    try:
        logs = service.get_logs(db=db)
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne lors de la récupération des logs.")