import logging
import time
from pythonjsonlogger import jsonlogger
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from database.database import engine
from database import models
from controllers import triage_controller
try:
    from core.logs import setup_logging          # Docker (WORKDIR = /app = contenu de api/)
except ImportError:
    from api.core.logs import setup_logging      # execution locale depuis la racine du projet

setup_logging()
logger = logging.getLogger("medical-chatbot-api")

models.Base.metadata.create_all(bind=engine)
app = FastAPI(title="Medical Triage API")

# ============================================================
# Métriques Prometheus
# ============================================================

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Middleware : log chaque requête HTTP
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency = round(time.time() - start, 3)
    logger.info(
        "http_request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "client_ip": request.client.host if request.client else "unknown",
        },
    )
    return response

app.include_router(triage_controller.router)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "message": "API is running"}
