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
    "Haute":   {"emoji": "🔴", "color": "error"},
    "Moyenne": {"emoji": "🟠", "color": "warning"},
    "Faible":  {"emoji": "🟢", "color": "success"},
}


def render_analyse(analyse_data: dict):
    """Affiche le verdict médical structuré avec couleurs."""
    urgence = analyse_data.get("urgence", "Faible")
    analyse = analyse_data.get("analyse", "")
    cfg = URGENCE_CONFIG.get(urgence, URGENCE_CONFIG["Faible"])

    if cfg["color"] == "error":
        st.error(f"{cfg['emoji']} Urgence **{urgence}**")
    elif cfg["color"] == "warning":
        st.warning(f"{cfg['emoji']} Urgence **{urgence}**")
    else:
        st.success(f"{cfg['emoji']} Urgence **{urgence}**")

    st.markdown(f"**Analyse médicale :** {analyse}")

    return f"{cfg['emoji']} Urgence **{urgence}**\n\n{analyse}"


def build_chatml(messages: list) -> list:
    """
    Construit la liste de messages au format ChatML (role/content)
    à partir de l'historique session_state.messages.
    Seuls les messages user/assistant sont envoyés (sans le system prompt,
    géré côté API/signatures.py).
    """
    chatml = []
    for msg in messages:
        if msg["role"] in ("user", "assistant"):
            chatml.append({"role": msg["role"], "content": msg["content"]})
    return chatml


# ==========================================
# ONGLET 1 : CHATBOT
# ==========================================
with tab_chat:

    # Initialisation de l'état de session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "triage_termine" not in st.session_state:
        st.session_state.triage_termine = False

    # Barre de contrôle
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("🔄 Nouveau patient", use_container_width=True):
            st.session_state.messages = []
            st.session_state.triage_termine = False
            st.rerun()

    st.markdown("---")

    # Zone de messages avec hauteur fixe pour ancrer le chat_input en bas
    chat_container = st.container(height=520, border=False)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Bandeau d'info si triage terminé
    if st.session_state.triage_termine:
        st.info(
            "✅ Triage terminé — le verdict a été rendu. "
            "Cliquez sur **🔄 Nouveau patient** pour démarrer une nouvelle consultation.",
            icon="🏁",
        )

    # Champ de saisie — désactivé après un verdict ANALYSE
    prompt = st.chat_input(
        "Décrivez vos symptômes ou ceux du patient…",
        disabled=st.session_state.triage_terme if hasattr(st.session_state, "triage_terme") else st.session_state.triage_termine,
    )

    if prompt and not st.session_state.triage_termine:

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Affichage immédiat du message utilisateur dans le container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Construction du contexte ChatML complet (tous les tours)
        chatml_messages = build_chatml(st.session_state.messages)

        # Appel API avec spinner dans le container
        with chat_container:
            with st.chat_message("assistant"):
                data = None
                error_msg = None
                with st.spinner("Analyse en cours…"):
                    try:
                        rep = requests.post(
                            f"{API_URL}/triage/ask",
                            json={"messages": chatml_messages},
                            timeout=60,
                        )
                        if rep.status_code == 200:
                            data = rep.json()
                        else:
                            error_msg = f"Erreur API ({rep.status_code}) : {rep.text}"
                    except requests.exceptions.Timeout:
                        error_msg = "L'API n'a pas répondu dans les temps (timeout 60s)."
                    except Exception as e:
                        error_msg = f"Impossible de joindre l'API à {API_URL}. Erreur : {e}"

                # Affichage hors spinner pour que le contenu reste visible
                if error_msg:
                    st.error(error_msg)

                elif data:
                    status = data.get("status")
                    latency = data.get("latency", 0.0)
                    st.caption(f"⏱️ Latence : **{latency} s**")

                    if status == "ANALYSE":
                        reponse_bot = render_analyse(data.get("data", {}))
                        st.session_state.messages.append(
                            {"role": "assistant", "content": reponse_bot}
                        )
                        # Verrouiller le chat : le triage est terminé
                        st.session_state.triage_termine = True
                        st.rerun()

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
