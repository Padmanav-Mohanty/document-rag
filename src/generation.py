from langchain_openrouter import ChatOpenRouter


def create_llm():
    """Create the LLM used to generate answers."""
    
    return ChatOpenRouter(
        model="inclusionai/ling-3.0-flash-vl:free",
        temperature=0,
    )