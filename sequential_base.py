"""
Sequential Base Pipeline Entrypoint
"""
from main import (
    PipelineState,
    editor_node,
    scriptwriter_node,
    translator_node,
    create_sequential_pipeline,
)

if __name__ == "__main__":
    import sys
    pipeline = create_sequential_pipeline()
    sample_text = (
        "Artificial Intelligence is changing the way we work, learn, and build software. "
        "Large language models like Gemini and GPT can understand natural language and "
        "generate human-like responses. But the real power of AI comes when we connect these "
        "models with tools, data, and external systems to create intelligent applications."
    )
    res = pipeline.invoke({"raw_input": sample_text})
    print(res["final_output"])
