from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from api.services import chatbot as service
from api.schemas import triage 
from api.database.database import get_db

router = APIRouter(prefix="/triage", tags=["Triage"])

@router.post("/ask", response_model=triage.TriageResponse)
def ask_triage(request: triage.TriageRequest, db: Session = Depends(get_db)):
    try:
        answer = service.generate_triage(request.symptomes)
        service.log_triage(db=db, question=request.symptomes, answer=answer)
        return {"triage": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne lors du triage.")