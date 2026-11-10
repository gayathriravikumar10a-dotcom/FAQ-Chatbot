# 🤖 FAQ Chatbot — Retrieval-Augmented Generation (RAG)

A semantic FAQ chatbot built using **Retrieval-Augmented Generation (RAG), Sentence Transformers, Google Gemini, and Streamlit**.

The application retrieves relevant information from a structured FAQ knowledge base and uses an LLM to generate context-grounded answers. It also includes a retrieval-based fallback mechanism to keep the chatbot useful when the Gemini API is temporarily unavailable.

---

## 🚀 Project Overview

Traditional FAQ systems often depend on exact keyword matching, which can make it difficult to understand questions expressed in different ways.

This project addresses that limitation using semantic search and RAG.

Users can ask questions in natural language. The application converts the question into an embedding, retrieves semantically similar FAQs, and uses the retrieved information as context for Gemini.

If the LLM is unavailable, the chatbot returns the most relevant retrieved FAQ answer instead.

## ✨ Key Features

- **Semantic Search:** Understands the meaning of user questions rather than relying only on exact keyword matching.
- **Retrieval-Augmented Generation:** Uses retrieved FAQ information as context for LLM responses.
- **Sentence Transformers:** Generates embeddings using `all-MiniLM-L6-v2`.
- **Cosine Similarity:** Ranks FAQs based on semantic similarity.
- **Gemini Integration:** Uses Gemini 3.1 Flash-Lite for answer generation.
- **Context-Grounded Responses:** Instructs the LLM to answer using the retrieved FAQ context.
- **Similarity Threshold:** Filters questions that are not sufficiently relevant to the FAQ knowledge base.
- **Fallback Mechanism:** Returns the best-matching FAQ answer when Gemini generation fails.
- **Source Attribution:** Displays the retrieved FAQ questions and similarity scores.
- **Streamlit Interface:** Provides an interactive chatbot interface.
- **Evaluation and Testing:** Includes scripts for retrieval evaluation, threshold analysis, and answer-grounding checks.

## 🏗️ System Architecture

```text
             USER
               |
               v
       Enter Natural Language
             Question
               |
               v
       Streamlit Interface
               |
               v
        FAQ RAG Pipeline
               |
               v
       Query Embedding
       (Sentence Transformers)
               |
               v
       Semantic Retrieval
        (Cosine Similarity)
               |
               v
       Similarity Threshold
               |
          +----+----+
          |         |
       Relevant   Not Relevant
          |         |
          v         v
    Retrieve Top-3  Rejection
       FAQs         Response
          |
          v
    Build FAQ Context
          |
          v
      Gemini LLM
          |
     +----+----+
     |         |
  Success    Failure
     |         |
     v         v
  Generated  Best-Matching
   Answer     FAQ Fallback
     |         |
     +----+----+
          |
          v
    Answer + Sources
          |
          v
         USER
```

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Gemini
