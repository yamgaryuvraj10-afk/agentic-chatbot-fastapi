from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import tempfile
import threading
import uuid
import gc

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from ai_agent import (
    get_response_from_ai_agent,
    extract_text_from_file,
    extract_tables_from_pdf,
    build_vectorstore_from_text_and_tables,
    answer_from_vectorstore,
)

class RequestState(BaseModel):
    session_id: str
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

class RagChatState(BaseModel):
    session_id: str
    model_name: str
    model_provider: str
    system_prompt: str
    query: str
    allow_search: bool

class UploadState(BaseModel):
    session_id: str

ALLOWED_MODEL_NAMES = [
    "llama3-70-8192",
    "mixtral-8x7b-32768",
    "llama-3.3-70b-versatile",
    "gpt-4o-mini",
]

app = FastAPI(title="LangGraph AI Agent")

_VECTORSTORES = {}
_JOBS = {}
_LOCK = threading.Lock()

def json_error(message: str, status_code: int = 400):
    return JSONResponse(status_code=status_code, content={"error": message})

@app.post("/chat")
def chat_endpoint(request: RequestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return json_error("Invalid model name. Kindly select a valid AI mode", 400)

    if not request.messages:
        return json_error("messages is empty", 400)

    try:
        answer = get_response_from_ai_agent(
            request.model_name,
            request.messages[-1],
            request.allow_search,
            request.system_prompt,
            request.model_provider,
        )
        return {"answer": answer}
    except Exception as e:
        return json_error(str(e), 500)

@app.post("/upload")
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    start_page: int = 1,
    end_page: int = 50,
):
    suffix = os.path.splitext(file.filename)[-1].lower()
    if suffix not in [".pdf", ".docx", ".txt"]:
        return json_error("Unsupported file type. Upload pdf, docx, or txt.", 400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id = str(uuid.uuid4())

    with _LOCK:
        _JOBS[job_id] = {
            "session_id": session_id,
            "status": "processing",
            "progress": 5,
            "message": "Upload received. Processing started.",
        }
        if session_id in _VECTORSTORES:
            del _VECTORSTORES[session_id]

    gc.collect()

    def worker():
        try:
            with _LOCK:
                _JOBS[job_id].update({"progress": 10, "message": "Extracting text..."})

            # Page-range applies only to PDFs; for docx/txt it is ignored
            text = extract_text_from_file(tmp_path, start_page=start_page, end_page=end_page)

            table_chunks = []
            if suffix == ".pdf":
                with _LOCK:
                    _JOBS[job_id].update({"progress": 25, "message": f"Extracting tables (pages {start_page}-{end_page})..."})

                table_chunks = extract_tables_from_pdf(
                    tmp_path,
                    start_page=start_page,
                    end_page=end_page,
                    jobs=_JOBS,
                    job_id=job_id,
                    lock=_LOCK,
                    max_workers=None,
                )

            if not text and not table_chunks:
                with _LOCK:
                    _JOBS[job_id].update({"status": "failed", "progress": 0, "message": "No readable text or tables found."})
                return

            with _LOCK:
                _JOBS[job_id].update({"progress": 86, "message": "Creating embeddings (text + tables)..."})

            vs = build_vectorstore_from_text_and_tables(
                text=text,
                table_chunks=table_chunks,
                job_id=job_id,
                jobs=_JOBS,
                lock=_LOCK,
            )

            with _LOCK:
                _VECTORSTORES[session_id] = vs
                _JOBS[job_id].update(
                    {"status": "done", "progress": 100, "message": "Indexing completed. You can use this document now."}
                )

        except Exception as e:
            with _LOCK:
                _JOBS[job_id].update({"status": "failed", "progress": 0, "message": str(e)})
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    threading.Thread(target=worker, daemon=True).start()
    return {"status": "accepted", "job_id": job_id}

@app.get("/upload_status")
def upload_status(job_id: str):
    with _LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return json_error("Invalid job_id", 404)
    return job

@app.post("/chat_rag")
def chat_rag(request: RagChatState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return json_error("Invalid model name. Kindly select a valid AI mode", 400)

    with _LOCK:
        vs = _VECTORSTORES.get(request.session_id)

    if vs is None:
        return json_error("No document uploaded for this session. Please upload a file first.", 400)

    try:
        result = answer_from_vectorstore(
            vs=vs,
            llm_id=request.model_name,
            provider=request.model_provider,
            system_prompt=request.system_prompt,
            query=request.query,
            allow_search=request.allow_search,
        )
        return result
    except Exception as e:
        return json_error(str(e), 500)

@app.post("/clear")
def clear_session(session: UploadState):
    with _LOCK:
        if session.session_id in _VECTORSTORES:
            del _VECTORSTORES[session.session_id]
    gc.collect()
    return {"status": "ok", "message": "Session document cleared."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
