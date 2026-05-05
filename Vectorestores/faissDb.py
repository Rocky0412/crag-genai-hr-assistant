from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma , FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core import documents
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
VECTOR_DIM = 384
RETRIEVE_TOP_K = 3
EMBED = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

#To store result
index=faiss.IndexFlatL2(VECTOR_DIM)
QA_CACHE = FAISS(
    embedding_function=EMBED,
    index=index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)

#To store Output
RAG_STORE = FAISS.from_texts(
    texts=[
        "LangGraph lets you compose stateful LLM workflows as graphs.",
        "In LangGraph, nodes can be cached; node caching memoizes outputs keyed by inputs for a TTL.",
        "Retrieval-Augmented Generation (RAG) retrieves external context and injects it into prompts.",
        "Semantic caching reuses prior answers when new questions are semantically similar."
    ],
    embedding=EMBED,
)

