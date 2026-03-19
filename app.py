import streamlit as st
from langchain_core.messages import HumanMessage
from main import workflow
from agent import ImageState

st.set_page_config(page_title="Story Teller 🧙", page_icon="📖", layout="centered")
st.title("📖 Story Teller Agent")
st.caption("Tell me what you want and I'll write you a story with an image! ✨")

if "state" not in st.session_state:
    st.session_state.state = ImageState(
        messages=[],
        user_input="",
        generation_output=None,
        story=None,
        generate_image=False
    )

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []

# ── Display previous conversation ─────────────────────────────
for entry in st.session_state.chat_display:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])
        if entry.get("image"):
            st.markdown(
                f'<img src="{entry["image"]}" width="512" style="border-radius:10px"/>',
                unsafe_allow_html=True
            )

# ── User input ────────────────────────────────────────────────
user_input = st.chat_input("Ask for a story or start a conversation...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_display.append({
        "role": "user",
        "content": user_input
    })

    st.session_state.state["user_input"] = user_input
    st.session_state.state["generation_output"] = None
    st.session_state.state["generate_image"] = False
    st.session_state.state["story"] = None

    with st.spinner("🧙 Crafting your story..."):
        st.session_state.state = workflow.invoke(st.session_state.state)

    ai_text = st.session_state.state.get("story") or ""
    image_data = st.session_state.state.get("generation_output")

    with st.chat_message("assistant"):
        if ai_text:
            st.write(ai_text)
        if image_data:
            st.markdown(
                f'<img src="{image_data}" width="512" style="border-radius:10px"/>',
                unsafe_allow_html=True
            )
            st.caption("🎨 Generated image for the story")
        elif st.session_state.state.get("generate_image"):
            st.warning("⏳ Image generation failed, try again.")

    st.session_state.chat_display.append({
        "role": "assistant",
        "content": ai_text,
        "image": image_data
    })