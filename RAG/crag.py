#uv run python -m RAG.crag
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma , FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from Models.grok import model
from Vectorestores.faissDb import QA_CACHE, RAG_STORE
from typing_extensions import TypedDict,Annotated,List,Optional
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
import time

class RAGState(TypedDict):
    question: str
    normalized_question: str
    context_docs: List[Document]
    answer: Optional[str]
    citations: List[str]
    cache_hit: bool

def normalized_question(state:RAGState)->RAGState:
  question=state["question"]
  normalized_question=question.strip().lower()
  state["normalized_question"]=normalized_question
  return state

def Semantic_checkup(state: RAGState) -> RAGState:
    normalized_question = state["normalized_question"]
    state["cache_hit"] = False

    hits = QA_CACHE.similarity_search_with_score(
        normalized_question,
        k=1
    )

    if not hits:
        return state  # no cache candidate

    doc, score = hits[0]

    # FAISS returns distance → lower is better
    THRESHOLD = 0.7

    if score < THRESHOLD:
        state["answer"] = doc.page_content
        state["cache_hit"] = True

    return state


def retrieve(state: RAGState) -> RAGState:
    q = state["normalized_question"]
    docs = RAG_STORE.similarity_search(q, k=2)
    state["context_docs"] = docs
    return state

def generate(state: RAGState) -> RAGState:
    q = state["question"]
    docs = state.get("context_docs", [])
    ctx = "\n\n".join([f"[doc-{i}] {d.page_content}" for i, d in enumerate(docs, start=1)])

    system = (
        "You are a precise RAG assistant. Use the context when helpful. "
        "Cite with [doc-i] markers if you use a fact from the context."
    )
    user = f"Question: {q}\n\nContext:\n{ctx}\n\nWrite a concise answer with citations."

    resp = model.invoke([{"role": "system", "content": system},
                       {"role": "user", "content": user}])
    state["answer"] = resp.content
    state["citations"] = [f"[doc-{i}]" for i in range(1, len(docs) + 1)]
    return state

def cache_write(state: RAGState) -> RAGState:
    q = state["normalized_question"]
    a = state.get("answer")
    if not q or not a:
        return state

    QA_CACHE.add_texts(
        texts=[q],
        metadatas=[{
            "answer": a,
            "ts": time.time(),
        }]
    )
    return state

#Intializeded graph
graph=StateGraph(RAGState)

graph.add_node("normalized_query",normalized_question)
graph.add_node("semantic_checkup",Semantic_checkup)
graph.add_node("retriver",retrieve)
graph.add_node("genarate",generate)
graph.add_node("cache_write",cache_write)


#add edges
graph.add_edge(START,"normalized_query")
graph.add_edge("normalized_query","semantic_checkup")
graph.add_conditional_edges("semantic_checkup",
                            lambda state: "end" if state["cache_hit"] else "retrieve",
    {
        "end":END,
        "retrieve": "retriver"
    }
   )
graph.add_edge("retriver", "genarate")
graph.add_edge("genarate", "cache_write")
graph.add_edge("cache_write", END)

memory=MemorySaver()

app=graph.compile(checkpointer=memory)

if __name__ == "__main__":
    thread_cfg = {"configurable": {"thread_id": "demo-user-1"}}

    q1 = "What is LangGraph ?"
    out1 = app.invoke({"question": q1, "context_docs": [], "citations": []}, thread_cfg)
    print("Answer:", out1["answer"])
    print("Citations:", out1.get("citations"))
    print("Cache hit?:", out1.get("cache_hit"))

