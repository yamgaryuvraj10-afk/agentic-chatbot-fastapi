import streamlit as st
import requests
import uuid
import pandas as pd

st.set_page_config(page_title="AI Agent", layout="wide")

API_BASE = "http://127.0.0.1:9999"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "ui_reset_key" not in st.session_state:
    st.session_state.ui_reset_key = 0

if "upload_job_id" not in st.session_state:
    st.session_state.upload_job_id = None

rk = st.session_state.ui_reset_key

MODEL_NAMES_GROQ = ["llama3-70-8192", "mixtral-8x7b-32768", "llama-3.3-70b-versatile"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

st.markdown(
    """
    <style>
      .block-container { padding-top: 0.35rem; padding-bottom: 1.0rem; max-width: 1300px; }
      header { visibility: hidden; height: 0px; }
      .title { font-size: 1.25rem; font-weight: 800; margin: 0.15rem 0 0.1rem 0; line-height: 1.2; }
      .subtle { color: rgba(255,255,255,0.7); font-size: 0.92rem; margin-top: 0.1rem; }
      .hr { height: 1px; background: rgba(120,120,120,0.20); margin: 0.5rem 0 0.7rem 0; }
      .toastwrap {
        position: fixed;
        right: 16px;
        bottom: 16px;
        z-index: 9999;
        width: 360px;
        max-width: calc(100vw - 32px);
      }
      .toast {
        border: 1px solid rgba(120,120,120,0.30);
        border-radius: 14px;
        padding: 12px 14px;
        background: rgba(20,20,20,0.92);
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
      }
      .toast h4 { margin: 0 0 6px 0; font-size: 0.95rem; }
      .toast p { margin: 0; color: rgba(255,255,255,0.75); font-size: 0.86rem; }
      .bar { margin-top: 10px; height: 10px; background: rgba(255,255,255,0.10); border-radius: 999px; overflow: hidden; }
      .bar > div { height: 100%; width: 0%; background: rgba(255,255,255,0.85); }
    </style>
    """,
    unsafe_allow_html=True
)

def show_progress_toast(progress: int, message: str):
    p = max(0, min(100, int(progress)))
    msg = message or "Processing..."
    st.markdown(
        f"""
        <div class="toastwrap">
          <div class="toast">
            <h4>Indexing document</h4>
            <p>{msg}</p>
            <div class="bar"><div style="width:{p}%;"></div></div>
            <p style="margin-top:8px;">{p}%</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def poll_job(job_id: str):
    try:
        r = requests.get(f"{API_BASE}/upload_status", params={"job_id": job_id}, timeout=5)
        return r.json()
    except Exception:
        return {"status": "failed", "progress": 0, "message": "Unable to fetch upload status."}

st.markdown('<div class="title">AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Banking-grade tables, page-range indexing, optional document RAG and web updates</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")

    provider = st.radio("Provider", ("Groq", "OpenAI"), key=f"provider_{rk}")

    if provider == "Groq":
        select_model = st.selectbox("Model", MODEL_NAMES_GROQ, key=f"model_groq_{rk}")
    else:
        select_model = st.selectbox("Model", MODEL_NAMES_OPENAI, key=f"model_openai_{rk}")

    allow_web_search = st.toggle("Enable web search", value=False, key=f"web_{rk}")
    use_rag = st.toggle("Use uploaded document (RAG)", value=False, key=f"rag_{rk}")

    system_prompt = st.text_area(
        "System prompt",
        height=90,
        placeholder="Define behavior and constraints",
        key=f"system_prompt_{rk}",
    )

    st.markdown("---")
    st.markdown("### Document")

    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        key=f"file_{rk}",
    )

    start_page = st.number_input("Start page", min_value=1, value=1, step=1)
    end_page = st.number_input("End page", min_value=1, value=50, step=1)

    c1, c2 = st.columns(2)
    with c1:
        upload_clicked = st.button("Upload", use_container_width=True, key=f"btn_upload_{rk}")
    with c2:
        clear_clicked = st.button("Clear", use_container_width=True, key=f"btn_clear_{rk}")

    reset_clicked = st.button("Reset session", use_container_width=True, key=f"btn_reset_{rk}")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")

if reset_clicked:
    try:
        requests.post(
            f"{API_BASE}/clear",
            json={"session_id": st.session_state.session_id},
            timeout=5,
        )
    except Exception:
        pass

    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.upload_job_id = None
    st.session_state.ui_reset_key += 1
    st.toast("Session reset", icon="✅")
    st.rerun()

if clear_clicked:
    r = requests.post(f"{API_BASE}/clear", json={"session_id": st.session_state.session_id})
    try:
        st.sidebar.success(r.json().get("message", "Cleared"))
    except Exception:
        st.sidebar.error("Clear API did not return JSON.")
        st.sidebar.code(r.text[:2000])

if upload_clicked:
    if uploaded_file is None:
        st.sidebar.error("Please select a file first.")
    else:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        params = {
            "session_id": st.session_state.session_id,
            "start_page": int(start_page),
            "end_page": int(end_page),
        }
        r = requests.post(f"{API_BASE}/upload", params=params, files=files)

        try:
            data = r.json()
        except Exception:
            st.sidebar.error("Upload API did not return JSON.")
            st.sidebar.code(r.text[:2000])
            st.stop()

        if "error" in data:
            st.sidebar.error(data["error"])
            st.stop()

        st.session_state.upload_job_id = data.get("job_id")
        st.toast("Indexing started", icon="⏳")
        st.rerun()

# Non-blocking polling
if st.session_state.upload_job_id:
    sd = poll_job(st.session_state.upload_job_id)
    if sd.get("status") == "done":
        st.toast("Indexing completed. You can query the document.", icon="✅")
        st.session_state.upload_job_id = None
    elif sd.get("status") == "failed":
        st.toast(f"Indexing failed: {sd.get('message')}", icon="❌")
        st.session_state.upload_job_id = None
    else:
        show_progress_toast(sd.get("progress", 0), sd.get("message", "Processing..."))
        if st.button("Refresh status"):
            st.rerun()

st.markdown("### Chat")

user_query = st.text_area(
    "Message",
    height=240,
    placeholder="Ask questions, or request charts like: 'Create a line chart for monthly revenue from the table'.",
    label_visibility="collapsed",
    key=f"query_{rk}",
)

ask_clicked = st.button("Send", key=f"btn_ask_{rk}")

if ask_clicked:
    if not user_query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Generating response..."):
            if use_rag:
                payload = {
                    "session_id": st.session_state.session_id,
                    "model_name": select_model,
                    "model_provider": provider,
                    "system_prompt": system_prompt,
                    "query": user_query,
                    "allow_search": allow_web_search,
                }
                response = requests.post(f"{API_BASE}/chat_rag", json=payload)
            else:
                payload = {
                    "session_id": st.session_state.session_id,
                    "model_name": select_model,
                    "model_provider": provider,
                    "system_prompt": system_prompt,
                    "messages": [user_query],
                    "allow_search": allow_web_search,
                }
                response = requests.post(f"{API_BASE}/chat", json=payload)

        try:
            data = response.json()
        except Exception:
            st.error("Chat API did not return JSON.")
            st.code(response.text[:2000])
            st.stop()

        if "error" in data:
            st.error(data["error"])
        else:
            st.markdown("#### Response")
            st.markdown(data.get("answer", ""))

            web = data.get("web", "")
            if web:
                with st.expander("Web updates"):
                    st.markdown(web)

            chart = data.get("chart")
            if isinstance(chart, dict) and chart.get("chart_type") and chart.get("x") and chart.get("series"):
                st.markdown("#### Chart")
                chart_type = (chart.get("chart_type") or "").lower()
                x = chart.get("x", [])
                series = chart.get("series", [])

                df = pd.DataFrame({"x": x})
                for s in series:
                    name = s.get("name", "Series")
                    y = s.get("y", [])
                    df[name] = y
                df = df.set_index("x")

                title = chart.get("title", "")
                if title:
                    st.caption(title)

                if chart_type == "bar":
                    st.bar_chart(df)
                elif chart_type in ["line", "area"]:
                    st.line_chart(df)
                elif chart_type in ["pie", "donut"]:
                    first = series[0]
                    sname = first.get("name", "Value")
                    pdf = pd.DataFrame({"label": x, sname: first.get("y", [])})
                    st.dataframe(pdf, use_container_width=True)
                    st.info("Pie chart display can be added via Plotly if required.")
                elif chart_type == "scatter":
                    first = series[0]
                    sname = first.get("name", "Value")
                    sdf = pd.DataFrame({"x": x, "y": first.get("y", [])})
                    st.dataframe(sdf, use_container_width=True)
                    st.info("Scatter chart display can be added via Plotly if required.")
                else:
                    st.warning("Unsupported chart type returned. Showing data table instead.")
                    st.dataframe(df, use_container_width=True)
