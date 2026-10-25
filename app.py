from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from agent import MSPARCAgent
from config import AgentConfig


st.set_page_config(page_title="M-SPARC Gemini Agent", layout="wide")
st.title("M-SPARC Natural Language Agent")
st.caption("Natural language → local M-SPARC knowledge → local MATLAB/M-SPARC → outputs and analysis")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Gemini API key",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        help="Used only when a request needs language-model reasoning. Routine run/read/sweep actions use local fast paths.",
    )
    workspace_text = st.text_input(
        "Simulation workspace",
        value=str(Path.cwd() / "workspace"),
        help="Folder containing writable M-SPARC cases and outputs.",
    )
    msparc_text = st.text_input(
        "M-SPARC repository",
        value=str(Path.cwd() / "M-SPARC-master"),
        help="Extracted M-SPARC repository containing src/, doc/, tests/, and psps/.",
    )
    model = st.selectbox(
        "Gemini model",
        options=["gemini-3.8-flash", "gemini-flash-lite-latest"],
        index=0,
        help="Flash is recommended for reliable structured case creation. Flash-Lite can be tried for lower latency.",
    )
    thinking_level = st.selectbox(
        "Thinking level",
        options=["low", "medium", "high"],
        index=0,
        help="Low is fastest. The agent only escalates low to medium for unusually complex case-generation prompts.",
    )

    if st.button("New conversation", use_container_width=True):
        if st.session_state.get("agent") is not None:
            st.session_state.agent.reset_conversation()
        st.session_state.chat_messages = []
        st.rerun()

    st.caption("M-SPARC syntax is parsed locally from doc/Manual.tex and official tests. Gemini never writes raw .inpt/.ion text directly.")

config_key = (workspace_text, msparc_text, model, thinking_level, api_key)
if st.session_state.get("config_key") != config_key:
    st.session_state.config_key = config_key
    st.session_state.agent = None
    st.session_state.chat_messages = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for image_path in message.get("images", []):
            if Path(image_path).exists():
                st.image(image_path)

prompt = st.chat_input("Tell the agent what to do with M-SPARC...")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if not api_key.strip():
                raise ValueError("Enter a Gemini API key in the sidebar or set GEMINI_API_KEY.")

            if st.session_state.agent is None:
                config = AgentConfig(
                    workspace_root=Path(workspace_text).expanduser().resolve(),
                    msparc_root=Path(msparc_text).expanduser().resolve(),
                    api_key=api_key.strip(),
                    model=model,
                    thinking_level=thinking_level,
                )
                st.session_state.agent = MSPARCAgent(config)

            with st.spinner("Working..."):
                answer = st.session_state.agent.chat(prompt)

            st.markdown(answer)
            image_paths = []
            for relative_path in st.session_state.agent.last_artifacts:
                absolute_path = Path(workspace_text).expanduser().resolve() / relative_path
                if absolute_path.suffix.lower() == ".png" and absolute_path.exists():
                    st.image(str(absolute_path))
                    image_paths.append(str(absolute_path))

            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "images": image_paths,
            })
        except Exception as exc:
            message = f"Configuration or runtime error: {exc}"
            st.error(message)
            st.session_state.chat_messages.append({"role": "assistant", "content": message, "images": []})
