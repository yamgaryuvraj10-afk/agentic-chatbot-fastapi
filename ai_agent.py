import os
import re
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import fitz  # PyMuPDF
import pandas as pd
import docx2txt

# -----------------------------
# LLM helpers
# -----------------------------
def _get_llm(provider: str, llm_id: str):
    if provider == "Groq":
        return ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    if provider == "OpenAI":
        return ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    raise ValueError("Invalid provider. Use Groq or OpenAI.")

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    llm = _get_llm(provider, llm_id)

    if allow_search:
        try:
            tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
            results = tool.invoke({"query": query})
        except Exception:
            results = []

        web_block = ""
        if isinstance(results, list) and results:
            lines = []
            for r in results[:3]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = (r.get("content", "") or "")[:350]
                lines.append(f"- {title}\n  {url}\n  {content}")
            web_block = "\n\nWeb results:\n" + "\n".join(lines)

        prompt = f"{system_prompt}\n\nUser:\n{query}\n{web_block}\n\nAnswer clearly."
        return llm.invoke(prompt).content

    prompt = f"{system_prompt}\n\nUser:\n{query}\n\nAnswer clearly."
    return llm.invoke(prompt).content


# -----------------------------
# Extraction: text
# -----------------------------
def extract_text_from_file(file_path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
    suffix = os.path.splitext(file_path)[-1].lower()

    if suffix == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if suffix == ".docx":
        return docx2txt.process(file_path) or ""

    if suffix == ".pdf":
        text_parts = []
        doc = fitz.open(file_path)
        total = doc.page_count
        sp = max(1, int(start_page))
        ep = int(end_page) if end_page else total
        ep = max(sp, min(ep, total))

        for i in range(sp - 1, ep):
            page = doc.load_page(i)
            t = page.get_text("text") or ""
            if t.strip():
                text_parts.append(f"[PAGE {i+1}]\n{t}")
        return "\n\n".join(text_parts)

    return ""


# -----------------------------
# Tables: digital + scanned, page-range, parallel scanned extraction
# -----------------------------
def _df_to_markdown(df: pd.DataFrame) -> str:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.fillna("")
    return df.to_markdown(index=False)

def _compress_markdown_table(md: str, head_rows: int = 20, tail_rows: int = 10) -> str:
    # Keep header + first N rows + last M rows. Prevent massive chunks.
    lines = [ln for ln in md.splitlines() if ln.strip()]
    if len(lines) <= (2 + head_rows + tail_rows):
        return md

    header = lines[:2]  # markdown header + separator
    body = lines[2:]
    if len(body) <= head_rows + tail_rows:
        return md

    kept = body[:head_rows] + ["| ... |" for _ in range(1)] + body[-tail_rows:]
    return "\n".join(header + kept)

def _camelot_tables_for_page(file_path: str, page_no_1based: int) -> List[str]:
    md_tables = []
    try:
        import camelot

        for flavor in ["lattice", "stream"]:
            try:
                tables = camelot.read_pdf(file_path, pages=str(page_no_1based), flavor=flavor)
                for t in tables:
                    df = t.df
                    if df is None or df.empty:
                        continue

                    df2 = df.copy()
                    df2.columns = df2.iloc[0].astype(str)
                    df2 = df2.drop(df2.index[0]).reset_index(drop=True)
                    if df2.shape[0] == 0 or df2.shape[1] == 0:
                        continue

                    md = _df_to_markdown(df2)
                    md_tables.append(_compress_markdown_table(md))
                if md_tables:
                    return md_tables
            except Exception:
                continue
    except Exception:
        return []
    return md_tables

def _safe_html_tables_to_markdown(html: str) -> List[str]:
    md_tables = []
    try:
        dfs = pd.read_html(html)
        for df in dfs:
            if df is not None and not df.empty:
                md_tables.append(_compress_markdown_table(_df_to_markdown(df)))
    except Exception:
        pass
    return md_tables

# Worker globals (per process)
_PPSTRUCT = None

def _ppstructure_init():
    global _PPSTRUCT
    try:
        from paddleocr import PPStructure
        _PPSTRUCT = PPStructure(show_log=False)
    except Exception:
        _PPSTRUCT = None

def _paddle_extract_tables_from_page(file_path: str, page_index_0based: int, dpi: int) -> List[str]:
    # Returns list of markdown tables, scanned extraction
    global _PPSTRUCT
    if _PPSTRUCT is None:
        return []

    try:
        import numpy as np
        import cv2

        doc = fitz.open(file_path)
        page = doc.load_page(page_index_0based)
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = _PPSTRUCT(img)
        md_tables = []

        for r in result:
            if r.get("type") == "table":
                res = r.get("res", {})
                html = ""
                if isinstance(res, dict):
                    html = res.get("html", "") or ""
                if not html and isinstance(res, str):
                    html = res

                if html:
                    md_tables.extend(_safe_html_tables_to_markdown(html))

        return md_tables
    except Exception:
        return []

def extract_tables_from_pdf(
    file_path: str,
    start_page: int,
    end_page: int,
    jobs=None,
    job_id=None,
    lock=None,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Returns list of:
    [{"page": int, "source": "camelot|paddle", "text": "markdown table"}]
    """
    doc = fitz.open(file_path)
    total = doc.page_count

    sp = max(1, int(start_page))
    ep = max(sp, min(int(end_page), total))

    pages = list(range(sp, ep + 1))
    table_chunks: List[Dict[str, Any]] = []

    # 1) Digital tables first (Camelot)
    pages_needing_scanned = []
    for idx, page_no in enumerate(pages, start=1):
        if jobs is not None and job_id and lock:
            with lock:
                jobs[job_id].update({
                    "progress": 25 + int((idx / max(len(pages), 1)) * 15),
                    "message": f"Extracting digital tables... page {page_no}/{ep}"
                })

        md_tables = _camelot_tables_for_page(file_path, page_no)
        if md_tables:
            for md in md_tables:
                table_chunks.append({"page": page_no, "source": "camelot", "text": md})
        else:
            pages_needing_scanned.append(page_no)

    # 2) Scanned tables (Paddle table structure) in parallel
    if pages_needing_scanned:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        cpu = os.cpu_count() or 4
        workers = max_workers if max_workers else max(2, min(cpu - 1, 6))

        ctx = mp.get_context("spawn")

        # Two-pass DPI strategy: 120 first, retry 160 only for pages with empty results
        def run_pass(dpi: int, target_pages: List[int]) -> Tuple[Dict[int, List[str]], List[int]]:
            found: Dict[int, List[str]] = {}
            retry_pages: List[int] = []

            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_ppstructure_init,
            ) as ex:
                futures = {}
                for p in target_pages:
                    futures[ex.submit(_paddle_extract_tables_from_page, file_path, p - 1, dpi)] = p

                done_count = 0
                for fut in as_completed(futures):
                    p = futures[fut]
                    done_count += 1

                    if jobs is not None and job_id and lock:
                        with lock:
                            # scanned extraction progress: 40..85
                            base = 40
                            span = 45
                            jobs[job_id].update({
                                "progress": base + int((done_count / max(len(target_pages), 1)) * span),
                                "message": f"Extracting scanned tables (dpi {dpi})... {done_count}/{len(target_pages)}"
                            })

                    md_tables = []
                    try:
                        md_tables = fut.result(timeout=1)
                    except Exception:
                        md_tables = []

                    if md_tables:
                        found[p] = md_tables
                    else:
                        retry_pages.append(p)

            return found, retry_pages

        found_120, retry = run_pass(120, pages_needing_scanned)
        found_160 = {}
        if retry:
            found_160, _ = run_pass(160, retry)

        for p, md_list in {**found_120, **found_160}.items():
            for md in md_list:
                table_chunks.append({"page": p, "source": "paddle", "text": md})

    return table_chunks


# -----------------------------
# Vector store build
# -----------------------------
def build_vectorstore_from_text_and_tables(
    text: str,
    table_chunks: List[Dict[str, Any]],
    job_id=None,
    jobs=None,
    lock=None,
):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    docs: List[Document] = []

    if text:
        parts = splitter.split_text(text)
        for p in parts:
            docs.append(Document(page_content=p, metadata={"type": "text"}))

    # table chunks: include metadata and compact content
    for t in table_chunks:
        content = f"TABLE | page={t['page']} | source={t['source']}\n{t['text']}"
        docs.append(Document(
            page_content=content,
            metadata={"type": "table", "page": t["page"], "source": t["source"]},
        ))

    if jobs is not None and job_id and lock:
        with lock:
            jobs[job_id].update({"progress": 88, "message": f"Creating embeddings for {len(docs)} chunks..."})

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True},
    )

    vs = FAISS.from_documents(docs, embeddings)

    if jobs is not None and job_id and lock:
        with lock:
            jobs[job_id].update({"progress": 97, "message": "Finalizing index..."})

    return vs


# -----------------------------
# Retrieval: rerank (optional but recommended)
# -----------------------------
def _try_rerank(query: str, docs: List[Document], top_k: int = 8) -> List[Document]:
    # If reranker not available, return original docs
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-base")
        pairs = [[query, d.page_content] for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:top_k]]
    except Exception:
        return docs[:top_k]


# -----------------------------
# Answer + Chart JSON
# -----------------------------
def _is_chart_request(query: str) -> bool:
    q = query.lower()
    keywords = ["chart", "graph", "plot", "bar", "line", "pie", "donut", "scatter", "histogram", "area"]
    return any(k in q for k in keywords)

def _safe_json_extract(s: str) -> Optional[dict]:
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def answer_from_vectorstore(
    vs,
    llm_id: str,
    provider: str,
    system_prompt: str,
    query: str,
    allow_search: bool,
) -> Dict[str, Any]:
    llm = _get_llm(provider, llm_id)

    # Retrieve more, then rerank
    retrieved = vs.similarity_search(query, k=20)
    docs = _try_rerank(query, retrieved, top_k=8)

    context = "\n\n".join([d.page_content for d in docs])

    web_block = ""
    if allow_search:
        try:
            tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
            results = tool.invoke({"query": query})
        except Exception:
            results = []
        if isinstance(results, list) and results:
            lines = []
            for r in results[:3]:
                title = r.get("title", "")
                url = r.get("url", "")
                content = (r.get("content", "") or "")[:350]
                lines.append(f"- {title}\n  {url}\n  {content}")
            web_block = "\n\nWeb updates (informational, may differ from your document):\n" + "\n".join(lines)

    base_guard = (
        "Rules:\n"
        "1) Answer ONLY from the provided CONTEXT.\n"
        "2) If the answer is not present in CONTEXT, say: Not found in the uploaded document.\n"
        "3) Do not guess numbers. Do not invent table values.\n"
    )

    if _is_chart_request(query):
        chart_prompt = (
            f"{system_prompt}\n\n"
            f"{base_guard}\n"
            "Task:\n"
            "User asked for a chart/graph. Use ONLY table values found in CONTEXT.\n"
            "Return JSON ONLY, no extra text.\n"
            "Schema:\n"
            "{\n"
            '  "chart_type": "bar|line|area|pie|scatter",\n'
            '  "title": "string",\n'
            '  "x_label": "string",\n'
            '  "y_label": "string",\n'
            '  "x": ["label1","label2"],\n'
            '  "series": [{"name":"Series 1","y":[1,2]}]\n'
            "}\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            f"USER QUERY:\n{query}\n"
        )

        raw = llm.invoke(chart_prompt).content
        chart = _safe_json_extract(raw)

        explain_prompt = (
            f"{system_prompt}\n\n"
            f"{base_guard}\n"
            "Provide a short explanation of what the chart represents, in 2 to 4 lines.\n\n"
            "CONTEXT:\n"
            f"{context}\n\n"
            f"USER QUERY:\n{query}\n"
        )
        explanation = llm.invoke(explain_prompt).content

        return {"answer": explanation, "chart": chart, "web": web_block}

    qa_prompt = (
        f"{system_prompt}\n\n"
        f"{base_guard}\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        f"USER QUERY:\n{query}\n\n"
        "Answer:"
    )
    answer = llm.invoke(qa_prompt).content
    return {"answer": answer, "chart": None, "web": web_block}
