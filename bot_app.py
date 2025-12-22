# bot_app.py
import os
import streamlit as st
from cli_bot import answer_question  # re-use function in cli_bot.py (so keep both in same folder)

st.set_page_config(page_title="ITFM RAG Chatbot", layout="wide")
st.title("IT Support Chatbot (Documents + Web)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Describe the IT problem (e.g., 'Printer jamming on Windows 10')")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("Searching history and web, then asking model..."):
        try:
            answer = answer_question(query)
        except Exception as e:
            answer = f"Error: {e}"
    st.session_state.history.append({"role": "assistant", "content": answer})

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
