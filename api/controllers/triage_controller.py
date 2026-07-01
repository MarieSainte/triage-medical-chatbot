from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from services import chatbot as service
from schemas import triage 
from database.database import get_db

router = APIRouter(prefix="/triage", tags=["Triage"])

@router.post("/ask", response_model=triage.TriageResponse)
def ask_triage(request: triage.TriageRequest, db: Session = Depends(get_db)):
    try:
        # Priorité aux messages ChatML, sinon fallback texte brut
        if request.messages:
            messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
            response_dict = service.generate_triage(messages=messages_dicts)
            # Log : on sérialise la conversation complète comme contexte
            import json
            contexte_log = json.dumps(messages_dicts, ensure_ascii=False)
        else:
            response_dict = service.generate_triage(symptomes=request.symptomes)
            import json
            contexte_log = request.symptomes or ""

        service.log_triage(db=db, question=contexte_log, answer=json.dumps(response_dict, ensure_ascii=False))
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