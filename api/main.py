from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine
from database import models
from controllers.triage_controller import triage

models.Base.metadata.create_all(bind=engine)

# Initialisation de l'API
app = FastAPI(title="Medical Triage API")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(triage.router)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "API is running"}