class Config:
    template_str = """
    You are very helpful assistant for question answering tasks. Use the pieces of retrieved context to answer question given. If you do not know 
    the answer, Just say that you do not know the answer instead of making up an answer.

    Retrieved context: {context}
    Query: {query}
    format instructions: {format_instructions}

    """

    default_google_model = "gemini-pro"
    default_openai_model = "gpt-3.5-turbo-0125"
    default_cohere_model = "command"
