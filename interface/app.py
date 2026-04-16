import streamlit as st
import requests
import os
import pandas as pd

st.set_page_config(page_title="Medical Chatbot Triage", page_icon="🏥", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🏥 Assistant Médical de Triage")

tab_chat, tab_logs = st.tabs(["💬 Chatbot", "📋 Logs (Base de données)"])

# ==========================================
# HELPERS
# ==========================================
URGENCE_CONFIG = {
    "Haute":   {"emoji": "🔴", "label": "HAUTE",   "color": "error"},
    "Moyenne": {"emoji": "🟠", "label": "MOYENNE",  "color": "warning"},
    "Faible":  {"emoji": "🟢", "label": "FAIBLE",   "color": "success"},
}

def render_analyse(analyse_data: dict):
    """Affiche le verdict médical structuré avec couleurs."""
    urgence = analyse_data.get("urgence", "Faible")
    analyse = analyse_data.get("analyse", "")

    cfg = URGENCE_CONFIG.get(urgence, URGENCE_CONFIG["Faible"])

    if cfg["color"] == "error":
        st.error(f"{cfg['emoji']} **Urgence {cfg['label']}** — {urgence}")
    elif cfg["color"] == "warning":
        st.warning(f"{cfg['emoji']} **Urgence {cfg['label']}** — {urgence}")
    else:
        st.success(f"{cfg['emoji']} **Urgence {cfg['label']}** — {urgence}")

    st.markdown(f"**Analyse médicale :** {analyse}")

    return f"{cfg['emoji']} **Urgence {cfg['label']}**\n\n{analyse}"


def build_context(messages: list) -> str:
    """Construit le contexte multi-tours pour l'API."""
    parts = []
    for msg in messages:
        role_tag = "Patient" if msg["role"] == "user" else "Médecin"
        parts.append(f"{role_tag} : {msg['content']}")
    return "\n\n".join(parts)


# ==========================================
# ONGLET 1 : CHATBOT
# ==========================================
with tab_chat:
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🔄 Nouveau patient", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Saisie utilisateur
    if prompt := st.chat_input("Décrivez vos symptômes ou ceux du patient…"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Contexte complet pour les conversations multi-tours
        contexte = build_context(st.session_state.messages)

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours…"):
                try:
                    rep = requests.post(
                        f"{API_URL}/triage/ask",
                        json={"symptomes": contexte},
                        timeout=30,
                    )

                    if rep.status_code == 200:
                        data = rep.json()
                        status = data.get("status")
                        latency = data.get("latency", 0.0)

                        st.caption(f"⏱️ Latence : **{latency} s**")

                        if status == "ANALYSE":
                            reponse_bot = render_analyse(data.get("data", {}))
                            st.session_state.messages.append(
                                {"role": "assistant", "content": reponse_bot}
                            )

                        elif status == "ASSISTANT":
                            reponse_bot = data.get(
                                "question",
                                "Pouvez-vous me donner plus de détails sur vos symptômes ?"
                            )
                            st.markdown(reponse_bot)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": reponse_bot}
                            )

                        else:
                            erreur_msg = data.get("message", "Erreur inconnue de l'IA.")
                            st.error(erreur_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": erreur_msg}
                            )

                    else:
                        st.error(f"Erreur API ({rep.status_code}) : {rep.text}")

                except requests.exceptions.Timeout:
                    st.error("L'API n'a pas répondu dans les temps (timeout 30s).")
                except Exception as e:
                    st.error(f"Impossible de joindre l'API à {API_URL}. Erreur : {e}")

# ==========================================
# ONGLET 2 : LOGS
# ==========================================
with tab_logs:
    st.subheader("Historique des triages (Base de données)")

    if st.button("🔄 Rafraîchir les logs"):
        try:
            res = requests.get(f"{API_URL}/triage/logs", timeout=10)
            if res.status_code == 200:
                logs_data = res.json()
                if logs_data:
                    df = pd.DataFrame(logs_data)
                    df["created_at"] = pd.to_datetime(df["created_at"])
                    df = df[["id", "created_at", "question", "answer"]]
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "id": "ID",
                            "created_at": st.column_config.DatetimeColumn(
                                "Date & Heure", format="DD/MM/YYYY HH:mm:ss"
                            ),
                            "question": "Contexte envoyé à l'IA",
                            "answer": "Réponse de l'IA",
                        },
                    )
                else:
                    st.info("Aucun log trouvé dans la base de données.")
            else:
                st.error("Erreur lors de la récupération des logs.")
        except Exception as e:
            st.error(f"Impossible de joindre l'API à {API_URL}. Erreur : {e}")
