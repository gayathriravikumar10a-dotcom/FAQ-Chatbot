
import streamlit as st
from chatbot.rag_pipeline import FAQRAG

st.set_page_config(
    page_title="GenAI FAQ Assistant",
    page_icon="💬",
    layout="centered",
)

@st.cache_resource
def load_rag():
    return FAQRAG()


st.title("💬 GenAI FAQ Assistant")
st.write(
    "Ask a question and get an answer grounded in "
    "your FAQ knowledge base using RAG and Gemini."
)

try:
    rag = load_rag()
except Exception as error:
    st.error(f"Could not initialize the RAG pipeline: {error}")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your AI FAQ assistant. "
                "Ask me about accounts, passwords, support, "
                "refunds, or other topics in my knowledge base."
            ),
            "sources": [],
        }
    ]


# Display the conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        if sources:
            with st.expander("View retrieved FAQ sources"):
                for source in sources:
                    st.write(f"**{source['question']}**")
                    st.caption(
                        f"Similarity score: {source['score']:.3f}"
                    )


# Accept a new question
user_question = st.chat_input("Ask your question...")

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
            "sources": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving FAQs and generating an answer..."):
            try:
                result = rag.answer_question(user_question)
                answer = result["answer"]
                sources = result.get("sources", [])

            except Exception:
                answer = (
                    "Sorry, I couldn't generate an answer right now. "
                    "Please try again in a moment."
                )
                sources = []
                st.warning(
                    "The RAG request failed. Check the VS Code "
                    "terminal for the error details."
                )

        st.markdown(answer)

        if sources:
            with st.expander("View retrieved FAQ sources"):
                for source in sources:
                    st.write(f"**{source['question']}**")
                    st.caption(
                        f"Similarity score: {source['score']:.3f}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )


with st.sidebar:
    st.header("About this project")
    st.write("**Architecture:** Retrieval-Augmented Generation")
    st.write("**Embeddings:** Sentence Transformers")
    st.write("**Retrieval:** Cosine similarity")
    st.write("**LLM:** Gemini")
    st.write("**Interface:** Streamlit")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
