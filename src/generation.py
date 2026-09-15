from langchain_openrouter import ChatOpenRouter


def create_llm():
    """Create the LLM used to generate answers."""
    
    return ChatOpenRouter(
        model="openai/120b:free",
        temperature=0,
    )