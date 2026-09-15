from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(retriever, llm):
    """Create a RAG chain that retrieves context and generates an answer."""

    prompt = ChatPromptTemplate.from_template(
        """Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)

        response = llm.invoke(
            prompt.format(
                context=context,
                question=question,
            )
        )

        return {
            "answer": response.content,
            "sources": docs,
        }

    return rag_chain