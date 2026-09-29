import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Load environment variables from .env
load_dotenv()

class PipelineState(TypedDict):
    """
    Represents the shared state that flows sequentially across the graph.
    Each node consumes and enriches this state dictionary.
    """
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str


# Initialize Groq LLM client
groq_api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

llm = ChatGroq(
    model=model_name,
    temperature=0.7,
    groq_api_key=groq_api_key,
)


def editor_node(state: PipelineState) -> dict:
    """
    Stage 1: Copyeditor Node
    Cleans up grammar, corrects punctuation/spelling mistakes, and refines
    transitions while keeping the core meaning intact.
    """
    print("\n--- [Stage 1] Executing Copyeditor Node ---")
    prompt = (
        "You are an expert copyeditor. Clean up the following raw text. "
        "Fix any grammatical errors, spelling mistakes, and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text.\n\n"
        f"Text:\n{state['raw_input']}"
    )
    response = llm.invoke(prompt)
    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: PipelineState) -> dict:
    """
    Stage 2: Scriptwriter Node
    Converts polished text into an engaging, conversational, punchy video script.
    """
    print("\n--- [Stage 2] Executing Scriptwriter Node ---")
    prompt = (
        "You are a charismatic YouTube content creator. Take this edited text and transform "
        "it into a highly engaging, punchy, conversational video script hook. Make it sound "
        "like a real person speaking passionately. Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )
    response = llm.invoke(prompt)
    return {"script_text": response.content.strip()}


def translator_node(state: PipelineState) -> dict:
    """
    Stage 3: Hinglish Localizer Node
    Adapts the script into natural, conversational Hinglish (Hindi + English blend)
    ideal for modern Indian audiences.
    """
    print("\n--- [Stage 3] Executing Hinglish Translator Node ---")
    prompt = (
        "You are an expert content localizer for the Indian market. Take the following script "
        "and convert it into natural, flowing 'Hinglish'. Do not simply translate it sentence "
        "by sentence or repeat information. Alternate comfortably between Hindi and English, as "
        "an intellectual tech educator would speak naturally on a live stream. "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )
    response = llm.invoke(prompt)
    return {"final_output": response.content.strip()}


def create_sequential_pipeline():
    """
    Builds and compiles the sequential LangGraph pipeline.
    Flow: START -> editor -> scriptwriter -> translator -> END
    """
    builder = StateGraph(PipelineState)

    # 1. Register nodes
    builder.add_node("editor", editor_node)
    builder.add_node("scriptwriter", scriptwriter_node)
    builder.add_node("translator", translator_node)

    # 2. Define sequential edges
    builder.add_edge(START, "editor")
    builder.add_edge("editor", "scriptwriter")
    builder.add_edge("scriptwriter", "translator")
    builder.add_edge("translator", END)

    # 3. Compile the graph
    return builder.compile()


if __name__ == "__main__":
    pipeline = create_sequential_pipeline()

    sample_input = (
        "Artificial Intelligence is changing the way we work, learn, and build software. "
        "Large language models like Gemini and GPT can understand natural language and "
        "generate human-like responses. But the real power of AI comes when we connect these "
        "models with tools, data, and external systems to create intelligent applications."
    )

    print("=" * 60)
    print("🚀 Running Sequential LangGraph Pipeline")
    print("=" * 60)
    print(f"\n[Raw Input]:\n{sample_input}\n")

    result = pipeline.invoke({"raw_input": sample_input})

    print("\n" + "=" * 60)
    print("✨ Pipeline Output Summary")
    print("=" * 60)
    print(f"\n[1. Edited Text]:\n{result.get('edited_text')}\n")
    print(f"[2. Script Text]:\n{result.get('script_text')}\n")
    print(f"[3. Final Hinglish Output]:\n{result.get('final_output')}\n")
