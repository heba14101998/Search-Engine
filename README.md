# FAQ Search Engine Project

This repository contains two Jupyter notebooks demonstrating different approaches to building a FAQ search engine using Python and modern NLP techniques.


## Project Structure
- `FAQ_basic_text_search.ipynb` — Basic text search with TF-IDF
- `FAQ_embeddings_vector_search.ipynb` — Embedding-based vector search (SVD, NMF, BERT)
- `README.md` — Project overview and instructions

## Notebooks

### 1. FAQ_basic_text_search.ipynb
A step-by-step guide to building a basic FAQ search engine using TF-IDF vectorization and cosine similarity.

**Key Steps:**
- Load FAQ data from a remote JSON file
- Vectorize text fields (`section`, `question`, `text`) using `TfidfVectorizer`
- Compute similarity between user queries and FAQ entries
- Apply boosting to prioritize certain fields (e.g., `question`)
- Filter results by course or other criteria
- Encapsulate logic in a reusable `TextSearch` class
- Example search with boosting and filtering

### 2. FAQ_embeddings_vector_search.ipynb
A notebook exploring advanced vector search techniques using embeddings, including SVD, NMF, and BERT.

**Key Steps:**
- Introduction to embeddings and their advantages over basic text search
- Load and preprocess FAQ data
- Generate embeddings using:
  - SVD (Singular Value Decomposition) on TF-IDF vectors
  - NMF (Non-Negative Matrix Factorization)
  - BERT (using Hugging Face Transformers)
- Perform semantic search by embedding queries and comparing with document embeddings
- Compare results from different embedding methods

## Requirements
- Python 3.8+
- Jupyter Notebook
- pandas, numpy, scikit-learn
- torch, transformers, tqdm (for BERT-based notebook)

Install dependencies with:
```bash
pip install pandas numpy scikit-learn torch transformers tqdm requests
```

## License
See [LICENSE](LICENSE) for details.

## Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [scikit-learn](https://scikit-learn.org/)
- [Original FAQ data source](https://github.com/alexeygrigorev/llm-rag-workshop)
