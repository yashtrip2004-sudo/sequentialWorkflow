import os
import sys
import streamlit as st
from dotenv import load_dotenv

# Ensure local imports work regardless of execution directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from main import create_sequential_pipeline

load_dotenv()

st.set_page_config(
    page_title="Sequential Base Pipeline",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Sequential Base Pipeline")
st.caption("A 3-stage LangGraph pipeline: Copyeditor ➔ Scriptwriter ➔ Hinglish Translator")

# Check for API Key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.warning("⚠️ `GROQ_API_KEY` not found in environment or .env file. Please configure it to run the pipeline.")

# Input area
default_text = (
    "Artificial Intelligence is changing the way we work, learn, and build software. "
    "Large language models like Gemini and GPT can understand natural language and "
    "generate human-like responses. But the real power of AI comes when we connect these "
    "models with tools, data, and external systems to create intelligent applications."
)

raw_input = st.text_area(
    "Enter Raw Text:",
    value=default_text,
    height=140,
    placeholder="Type or paste text here..."
)

if st.button("Run Pipeline", type="primary", use_container_width=True):
    if not raw_input.strip():
        st.error("Please provide some input text before running.")
    else:
        with st.spinner("Executing sequential graph stages..."):
            try:
                pipeline = create_sequential_pipeline()
                result = pipeline.invoke({"raw_input": raw_input})

                st.success("Pipeline executed successfully!")

                st.subheader("1. Copyeditor Output")
                st.write(result.get("edited_text", ""))

                st.divider()

                st.subheader("2. Scriptwriter Output")
                st.write(result.get("script_text", ""))

                st.divider()

                st.subheader("3. Hinglish Translator Output")
                st.write(result.get("final_output", ""))

            except Exception as e:
                st.error(f"Error executing pipeline: {e}")
