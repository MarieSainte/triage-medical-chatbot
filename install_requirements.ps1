# Installation des packages
uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

# Modèle léger pour tes symptômes
uv run python -m spacy download en_core_web_sm

# Modèles "Medium" OBLIGATOIRES pour l'anonymisation Presidio
uv run python -m spacy download fr_core_news_md
uv run python -m spacy download en_core_web_md