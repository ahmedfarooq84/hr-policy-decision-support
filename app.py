import os
import uuid
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from pypdf import PdfReader
import chromadb
import pandas as pd
import numpy as np
import time
import shutil
import stat
# PDF generation for escalation memo
from fpdf import FPDF
from datetime import datetime

# Optional OpenAI client (only used if API key exists)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


APP_TITLE = "Intelligent HR Decision Support System (HR-DSS)"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hr_policies"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120

# ---- Retrieval gate ----
# Chroma returns cosine distances (lower is better). Similarity ~= (1 - distance).
# If results are below this similarity threshold, treat as 'no retrieval'.
SIMILARITY_THRESHOLD = 0.15


@dataclass
class Chunk:
    text: str
    source: str
    page: int


# ---------------- API / RAG Helpers ----------------
# --- AUTO-LOADER FOR DEMO (Add this after imports) ---
def load_demo_data():
    """Forces the demo files to load if the DB is empty."""
    
    # 1. Define your exact file names
    demo_files = [
        "Extended_Remote_Work_and_International_Assignment_Policy.pdf",
        "Extended_Code_of_Conduct_and_Termination_Guidelines.pdf",
        "Extended_Policy_Gaps_and_Legal_Escalation_Guidelines.pdf"
    ]
    
    # 2. Check if we already have data. If yes, stop.
    if "local_chunks" in st.session_state and len(st.session_state.local_chunks) > 0:
        return

    # 3. If no data, load the files automatically!
    total_chunks = 0
    st.toast("⚙️ Auto-loading Demo Policies... Please wait.")
    
    for fname in demo_files:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                # We reuse your existing index_pdf function
                file_bytes = f.read()
                # Determine mode (Semantic or Keyword) based on API key presence
                # Note: index_pdf uses st.session_state internally
                index_pdf(fname, file_bytes)
                total_chunks += 1
    
    if total_chunks > 0:
        st.toast(f"✅ Ready! Loaded {len(demo_files)} demo policies.")

def _rmtree_force(path: str, retries: int = 6, delay: float = 0.4):
    def onerror(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    for i in range(retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, onerror=onerror)
            return
        except PermissionError:
            time.sleep(delay)
    # last try (raise if still locked)
    if os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)

def get_api_key() -> str:
    """Safely read OPENAI_API_KEY from sidebar override, Streamlit secrets, or environment."""
    override = (st.session_state.get("api_key_override", "") or "").strip()
    if override:
        return override

    # Streamlit Cloud: Secrets
    try:
        secret = (st.secrets.get("OPENAI_API_KEY", "") or "").strip()
        if secret:
            return secret
    except Exception:
        pass

    # Local env
    return (os.getenv("OPENAI_API_KEY") or "").strip()


def api_available() -> bool:
    return bool(get_api_key()) and (OpenAI is not None)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= n:
            break

        start = end - overlap

    return chunks


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_openai(texts: List[str]) -> List[List[float]]:
    api_key = get_api_key()
    if not api_key or OpenAI is None:
        raise RuntimeError("OPENAI_API_KEY is not set (embeddings disabled).")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


def semantic_search(query: str, k: int = 15) -> List[Chunk]:
    if not api_available():
        return []

    collection = get_collection()
    q_emb = embed_openai([query])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    out: List[Chunk] = []
    docs = (res.get("documents", [[]])[0] or [])
    metas = (res.get("metadatas", [[]])[0] or [])
    dists = (res.get("distances", [[]])[0] or [])

    # ---- Threshold filter: if similarity is too low, treat as "no retrieval" ----
    for d, m, dist in zip(docs, metas, dists):
        try:
            similarity = 1.0 - float(dist)
        except Exception:
            similarity = 0.0

        if similarity < SIMILARITY_THRESHOLD:
            continue

        out.append(Chunk(text=d, source=m.get("source", "Unknown"), page=int(m.get("page", 0))))
    return out


def keyword_search(query: str, chunks: List[Chunk], k: int = 5) -> List[Chunk]:
    q = query.lower().strip()
    if not q:
        return []
    terms = [t for t in q.split() if len(t) >= 3]

    scored: List[Tuple[int, Chunk]] = []
    for c in chunks:
        t = (c.text or "").lower()
        score = sum(t.count(term) for term in terms)
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


def unique_source_pages(retrieved: List[Chunk]) -> List[Tuple[str, int]]:
    seen = set()
    out: List[Tuple[str, int]] = []
    for c in retrieved:
        key = (c.source, c.page)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# ---------------- Confidence (CONSISTENT) ----------------
# ---------------- Confidence (DEMO TUNED) ----------------
def confidence_from_unique_sources(n_unique: int) -> str:
    if n_unique >= 2:  # Lowered from 4 to 2 matches your 2-page PDFs
        return "High"
    if n_unique == 1:
        return "Medium"
    return "Low"

def confidence_badge(n_unique: int) -> str:
    if n_unique >= 2:
        return "🟢 High (multiple policy citations)"
    if n_unique == 1:
        return "🟡 Medium (limited policy citations)"
    return "🔴 Low (no policy citations)"


def confidence_meter(n_unique: int) -> Tuple[float, str]:
    if n_unique >= 2:
        return 0.90, "High"
    if n_unique == 1:
        return 0.50, "Medium"
    return 0.05, "Low"

def risk_banner(n_unique: int):
    if n_unique >= 2:
        st.success("✅ High confidence: Policy is clear and grounded.")
    elif n_unique == 1:
        st.warning("⚠️ Medium confidence: Only one source found. Review recommended.")
    else:
        st.error("🚫 Low confidence: No relevant policy found. Escalation required.")


def index_pdf(file_name: str, file_bytes: bytes) -> int:
    # Clear the old "No policy found" state instantly
    st.session_state.last_retrieved = []
    st.session_state.last_answer = ""
    st.session_state.last_n_unique = 0
    # Clear old results to force a fresh search when a new file arrives
    if "last_retrieved" in st.session_state:
        st.session_state.last_retrieved = []
    if "last_answer" in st.session_state:
        st.session_state.last_answer = ""
    """Parse PDF → chunk → store in Chroma (semantic) AND session_state (keyword fallback)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    chunks: List[Chunk] = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if not text.strip():
            continue

        for piece in chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(Chunk(text=piece, source=file_name, page=i + 1))

    # ✅ ALWAYS keep local chunks for keyword fallback (industry standard)
    st.session_state["local_chunks"] = st.session_state.get("local_chunks", []) + chunks

    docs = [c.text for c in chunks]
    metas = [{"source": c.source, "page": c.page} for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    if api_available():
        collection = get_collection()
        embs = embed_openai(docs)
        collection.add(documents=docs, metadatas=metas, ids=ids, embeddings=embs)
        st.success(f"Indexed {len(chunks)} chunks (semantic embeddings enabled).")
    else:
        st.warning("No API key detected. Running in Retrieval-Only mode (keyword search).")
        st.success(f"Indexed {len(chunks)} chunks (local keyword index).")

    return len(chunks)



def build_prompt(question: str, retrieved: List[Chunk]) -> str:
    blocks = []
    for i, c in enumerate(retrieved, start=1):
        blocks.append(f"[Evidence {i}] {c.source} (Page {c.page})\n{c.text}")
    context = "\n\n".join(blocks)

    return f"""
You are an HR decision support assistant.

Rules:
- Use ONLY the evidence provided under "Evidence".
- If the evidence is insufficient, say exactly: "I don't have enough policy context to answer that."
- Do NOT guess or invent policy.
- Keep the answer short, clear, and actionable.
- Do NOT include a "Sources:" line. The app will show sources separately.

Question:
{question}

Evidence:
{context}
""".strip()


def answer_openai(prompt: str) -> str:
    api_key = get_api_key()
    if not api_key or OpenAI is None:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are careful, policy-grounded, and risk-aware."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ---------------- Dashboard (Synthetic Demo Data) ----------------
def make_synthetic_workforce_data(n=600, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    levels = rng.choice(
        ["IC1", "IC2", "IC3", "M1", "M2", "Dir"],
        size=n,
        p=[0.18, 0.26, 0.24, 0.18, 0.10, 0.04],
    )
    regions = rng.choice(
        ["US", "Canada", "India", "UK", "Germany"],
        size=n,
        p=[0.55, 0.08, 0.22, 0.08, 0.07],
    )
    tenure_yrs = np.clip(rng.normal(4.2, 2.8, size=n), 0, 15)

    base_map = {"IC1": 90000, "IC2": 115000, "IC3": 145000, "M1": 170000, "M2": 210000, "Dir": 260000}
    base = np.array([base_map[l] for l in levels]) * rng.normal(1.0, 0.08, size=n)

    level_risk = {"IC1": 0.06, "IC2": 0.05, "IC3": 0.04, "M1": 0.035, "M2": 0.03, "Dir": 0.02}
    p_attr = np.array([level_risk[l] for l in levels]) + np.clip(0.06 - 0.01 * tenure_yrs, 0.0, 0.06)
    p_attr = np.clip(p_attr, 0.01, 0.12)
    attrited = rng.binomial(1, p_attr)

    return pd.DataFrame(
        {
            "level": levels,
            "region": regions,
            "tenure_yrs": tenure_yrs.round(1),
            "base_salary_usd": base.round(0).astype(int),
            "attrited": attrited,
        }
    )


def make_attrition_trend(df: pd.DataFrame, months=12, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_rate = float(df["attrited"].mean())
    noise = rng.normal(0, 0.003, size=months)
    season = 0.004 * np.sin(np.linspace(0, 2 * np.pi, months))
    rates = np.clip(baseline_rate + noise + season, 0.01, 0.15)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="MS")
    return pd.DataFrame({"month": idx, "attrition_rate": rates})


# ---------------- Escalation Memo (PDF) ----------------
def create_escalation_pdf(
    question: str,
    risk_assessment: str,
    decision_summary: str,
    source_docs: List[Tuple[str, int]],
    top_snippets: List[str],
    n_unique_sources: int,
) -> bytes:

    def _latin1_safe(text: str) -> str:
        if text is None:
            return ""
        return str(text).encode("latin-1", "replace").decode("latin-1")

    def _risk_stamp_from_unique_sources(n_unique: int):
        if n_unique >= 4:
            return "LOW RISK", (20, 150, 60)
        if n_unique >= 2:
            return "MEDIUM RISK", (230, 170, 30)
        return "HIGH RISK", (200, 60, 60)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Background ----
    try:
        pdf.image("memo_bg.jpg", x=0, y=0, w=210, h=297)
    except Exception:
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "INTERNAL MEMORANDUM (Background image missing)", ln=True, align="C")

    # ---- Layout constants ----
    LEFT = 25
    TOP = 58
    RIGHT = 20

    pdf.set_left_margin(LEFT)
    pdf.set_right_margin(RIGHT)
    pdf.set_xy(LEFT, TOP)

    # ---- Meta Section ----
    case_id = uuid.uuid4().hex[:8].upper()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf.set_font("Arial", "B", 12)
    pdf.set_x(LEFT)
    pdf.cell(0, 8, _latin1_safe(f"Case Reference: #{case_id}"), ln=True)

    pdf.set_font("Arial", "", 11)
    pdf.set_x(LEFT)
    pdf.cell(0, 7, _latin1_safe(f"Generated: {now}"), ln=True)

    # Save Y position after meta block (prevents cursor drift from stamp)
    meta_y_end = pdf.get_y()

    # ---- Risk Stamp ----
    label, (r, g, b) = _risk_stamp_from_unique_sources(n_unique_sources)
    stamp_w, stamp_h = 40, 10
    stamp_x = 210 - RIGHT - stamp_w
    stamp_y = TOP - 3

    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.rect(stamp_x, stamp_y, stamp_w, stamp_h, style="F")
    pdf.set_xy(stamp_x, stamp_y + 2)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(stamp_w, 6, label, align="C")

    pdf.set_text_color(0, 0, 0)

    # Return to left margin under meta block
    pdf.set_xy(LEFT, meta_y_end + 6)

    # ---- Helpers ----
    def section_title(title: str):
        pdf.set_x(LEFT)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, _latin1_safe(title), ln=True)
        pdf.set_font("Arial", "", 11)

    def body(text: str, line_h: float = 6.0):
        pdf.set_x(LEFT)
        pdf.multi_cell(0, line_h, _latin1_safe(text))

    # ---- Content Sections ----
    section_title("Employee Query:")
    body(question)

    pdf.ln(2)
    section_title("AI Risk Assessment:")
    body(risk_assessment)

    pdf.ln(2)
    section_title("Decision Summary (Decision Support Output):")
    body(decision_summary)

    pdf.ln(2)
    section_title("Sources Reviewed:")
    if not source_docs:
        body("No policy sources were retrieved.")
    else:
        for (src, pg) in source_docs:
            body(f"- {src} (Page {pg})", line_h=5.5)

    pdf.ln(2)
    section_title("Top Retrieved Snippets (for Legal Review):")
    if not top_snippets:
        body("[No snippets found]")
    else:
        for i, s in enumerate(top_snippets, start=1):
            pdf.set_x(LEFT)
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 6, _latin1_safe(f"[Snippet {i}]"), ln=True)
            pdf.set_font("Arial", "", 9)
            pdf.set_x(LEFT)
            pdf.multi_cell(0, 5.0, _latin1_safe(s))
            pdf.ln(1)

    out = pdf.output(dest="S")  # fpdf2 returns bytearray
    return bytes(out)


# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")

# Banner image at the very top (local file)
# Make sure the file name matches exactly (extensions matter on deploy).
st.image("HR-DSS.jpg", use_container_width=True)

st.title(APP_TITLE)
st.caption("Policy-grounded HR decision support (RAG). Built for a safe, impressive live demo.")

st.info(
    "⚠️ **Decision Support Only**  \n"
    "This tool assists HR decision-making but does **not** replace HR or Legal judgment.  \n"
    "Answers are generated **only from retrieved policy documents**.  \n"
    "If confidence is Medium or Low, escalate before taking action."
)

# Session state
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "local_chunks" not in st.session_state:
    st.session_state.local_chunks = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = "Low"
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_n_unique" not in st.session_state:
    st.session_state.last_n_unique = 0


# ---------------- Sidebar: ALL SETTINGS ----------------
with st.sidebar:
    st.header("Settings")

    st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        key="api_key_override",
        help="If provided, semantic embeddings + LLM answering are enabled. Otherwise the app runs in keyword retrieval-only mode.",
    )
    st.write(f"LLM Mode: {'✅ ON' if api_available() else '❌ OFF (Retrieval-Only)'}")

    st.divider()

    st.subheader("Upload Policy PDFs")
    uploaded = st.file_uploader("Upload one or more HR policy PDFs", type=["pdf"], accept_multiple_files=True)

    if uploaded:
        total = 0
        for f in uploaded:
            total += index_pdf(f.name, f.getvalue())
        st.success(f"Processed files. Total chunks handled: {total}")

    st.divider()

    st.divider()

    if st.button("Reset Index", type="secondary"):
    try:
        # Clear session state
        st.session_state.local_chunks = []
        st.session_state.audit_log = []
        st.session_state.last_retrieved = []
        st.session_state.last_question = ""
        st.session_state.last_answer = ""
        st.session_state.last_n_unique = 0

        # Clear Chroma collection without deleting files (avoids WinError 32)
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass  # collection may not exist yet

        st.success("Reset complete. The index has been cleared.")
        time.sleep(0.5)
        st.rerun()

    except Exception as e:
        st.error(f"Reset failed: {e}. Use 'Reboot App' as a fallback.")


    


# ---------------- Main Area ----------------
load_demo_data()
chat_col, dash_col = st.columns([1.25, 0.95], gap="large")

with chat_col:
    st.subheader("Policy Chat (Grounded)")

    st.markdown("#### Demo Scenarios")
    s1, s2, s3 = st.columns(3)
    if s1.button("Remote Work Policy"):
        st.session_state["q"] = "What is our policy on remote work from another Country?"
    if s2.button("Termination due to Conduct"):
        st.session_state["q"] = "What is the process for termination due to conduct?"
    if s3.button("Pet Iguana (Out of Scope)"):
        st.session_state["q"] = "Can I bring my pet iguana to the office?"

# Row 2: New Standby Policy Scenarios
    r2_s1, r2_s2, r2_s3 = st.columns(3)
    if r2_s1.button("International Business Class"):
        st.session_state["q"] = "What is the policy for booking international business class flights?"
    if r2_s2.button("Ownership of Ideas"):
        st.session_state["q"] = "Who owns the ideas I come up with at work?"
    if r2_s3.button("Political Hats (Out of Scope)"):
        st.session_state["q"] = "What is the company policy on wearing political hats?"

    question = st.text_input("Ask a question:", key="q", placeholder="Type an HR policy question…")
    ask = st.button("Get Answer", type="primary")

    if ask and question.strip():
        if not api_available():
             st.caption("🔎 Retrieval mode: Keyword fallback (no valid OpenAI key detected).")
        else:
             st.caption("🔎 Retrieval mode: Semantic + Keyword fallback.")
        # 1. Try Smart Vector Search first     
        retrieved = semantic_search(question, k=15)
        if not retrieved and not api_available():
            retrieved = keyword_search(question, st.session_state.local_chunks, k=5)

        st.session_state.last_retrieved = retrieved
        st.session_state.last_question = question

        uniq = unique_source_pages(retrieved)
        n_unique = len(uniq)
        st.session_state.last_n_unique = n_unique
        st.session_state.last_confidence = confidence_from_unique_sources(n_unique)

        # audit
        if retrieved:
            st.session_state.audit_log.insert(
                0,
                {
                    "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question,
                    "confidence": st.session_state.last_confidence,
                    "n_sources": n_unique,
                    "sources": [{"source": c.source, "page": c.page, "chunk": (c.text or "")[:350]} for c in retrieved],
                },
            )

        # Confidence meter
        meter_val, meter_label = confidence_meter(n_unique)
        st.markdown(f"**Confidence:** {confidence_badge(n_unique)}")
        st.progress(meter_val, text=f"Confidence Meter: {meter_label} ({int(meter_val*100)}%)")
        risk_banner(n_unique)

        with st.expander("Safety Notes (Responsible AI)"):
            st.markdown(
                """
- This system is a **decision support tool**, not an automated decision-maker.
- Answers are generated **only from retrieved policy context** (RAG grounding).
- If confidence is Medium/Low, the user should **escalate to HR/Legal** before action.
- All outputs are **auditable** (question, retrieved sources, response).
"""
            )

        if not retrieved:
            st.warning("No relevant policy sections found. Upload a policy PDF or try a more specific query.")
            st.session_state.last_answer = ""
        else:
            st.markdown("### Answer")
            if api_available():
                prompt = build_prompt(question, retrieved)
                answer = answer_openai(prompt)
                st.session_state.last_answer = answer
                st.write(answer)
            else:
                st.session_state.last_answer = ""
                st.write("Retrieval-only mode: showing the most relevant policy sections below.")

            st.markdown("### Sources Used (Top Matches)")
            seen = set()
            idx = 1
            for c in retrieved:
                key = (c.source, c.page)
                if key in seen:
                    continue
                seen.add(key)
                with st.expander(f"Source {idx}: {c.source} — Page {c.page}"):
                    st.write(c.text)
                idx += 1

    # Escalate to Legal feature
    st.divider()
    st.subheader("Escalation")

    # ENABLE button even if no sources found (Critical for Red Light scenarios)
    can_escalate = bool(st.session_state.get("last_question"))
    if st.button("Escalate to Legal", disabled=not can_escalate):
        q = st.session_state.get("last_question", "").strip()
        conf_label = st.session_state.get("last_confidence", "Low")
        retrieved = st.session_state.get("last_retrieved", [])
        answer = (st.session_state.get("last_answer", "") or "").strip()
        n_unique = int(st.session_state.get("last_n_unique", 0))

        uniq = unique_source_pages(retrieved)
        decision_summary = answer if answer else "[No LLM answer generated — retrieval-only mode]"

        if n_unique >= 4:
            risk_text = "Low interpretive risk. Policy coverage is strong."
        elif n_unique >= 2:
            risk_text = "Moderate interpretive risk. Limited policy sections retrieved. Recommend Legal review before action."
        else:
            risk_text = "High interpretive risk due to limited policy context. Legal review recommended."

        risk_assessment = f"System Confidence: {conf_label}\n\n{risk_text}"

        src_lines = [f"- {src} (Page {pg})" for (src, pg) in uniq] if uniq else ["- None"]

        # Top snippets: 2–3 unique source/page snippets
        seen = set()
        top_snippets = []
        for c in retrieved:
            key = (c.source, c.page)
            if key in seen:
                continue
            seen.add(key)
            snippet = (c.text or "").strip()[:900]
            top_snippets.append(f"{c.source} (Page {c.page})\n{snippet}")
            if len(top_snippets) >= 3:
                break

        escalation_message = f"""
Subject: HR Policy Escalation Request — Guidance Needed

Hi Legal Team,

I’m requesting guidance on an HR policy interpretation question before taking action.

Question:
{q}

Risk Assessment:
{risk_assessment}

Policy Sources Retrieved:
{chr(10).join(src_lines)}

Decision Summary (Decision Support Output):
{decision_summary}

Top Retrieved Snippets:
{chr(10).join([f"[Snippet {i+1}] {s}" for i, s in enumerate(top_snippets)]) if top_snippets else "[No snippets found]"}

Please advise on the correct interpretation and any required steps/approvals.

Thank you,
[Your Name]
""".strip()

        st.success("Escalation draft generated.")
        st.code(escalation_message)

        pdf_bytes = create_escalation_pdf(
            question=q,
            risk_assessment=risk_assessment,
            decision_summary=decision_summary,
            source_docs=uniq,
            top_snippets=top_snippets,
            n_unique_sources=n_unique,
        )

        st.download_button(
            label="Download Escalation Memo (PDF)",
            data=pdf_bytes,
            file_name="HR_Escalation_Memo.pdf",
            mime="application/pdf",
        )

    st.caption("Tip: Use escalation especially when confidence is 🟡 or 🟠 / 🔴.")

    st.divider()
    st.subheader("Audit Log (Demo)")
    if len(st.session_state.audit_log) == 0:
        st.caption("No queries yet. Ask a policy question to populate the audit trail.")
    else:
        latest = st.session_state.audit_log[0]
        st.write(f"**Latest:** {latest['time']}")
        st.write(f"**Question:** {latest['question']}")
        st.write(f"**Confidence:** {latest.get('confidence', 'Low')}")
        st.write(f"**Sources used:** {latest['n_sources']}")

        with st.expander("View retrieved policy snippets"):
            seen = set()
            idx = 1
            for s in latest.get("sources", []):
                key = (s.get("source"), s.get("page"))
                if key in seen:
                    continue
                seen.add(key)
                st.markdown(f"**{idx}. {s.get('source', 'Unknown')} (p.{s.get('page', '?')})**")
                chunk = (s.get("chunk") or "").strip()
                st.code(chunk if chunk else "[No snippet captured]")
                idx += 1

        with st.expander("Full Audit Log"):
            for entry in st.session_state.audit_log[:10]:
                st.markdown(
                    f"**{entry['time']}** — {entry['question']}  \n"
                    f"Confidence: {entry.get('confidence', 'Low')}  \n"
                    f"Sources: {entry['n_sources']}"
                )


with dash_col:
    st.subheader("Workforce Snapshot (Demo)")

    c1, c2, c3 = st.columns(3)
    with c1:
        seed = st.number_input("Synthetic seed", min_value=1, max_value=9999, value=42, step=1)
    with c2:
        n = st.slider("Employees", min_value=200, max_value=2000, value=600, step=50)
    with c3:
        view = st.selectbox("View", ["Overview", "By Level", "By Region"])

    df = make_synthetic_workforce_data(n=int(n), seed=int(seed))
    trend = make_attrition_trend(df, months=12, seed=int(seed) + 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Employees", f"{len(df):,}")
    k2.metric("Attrition (synthetic)", f"{df['attrited'].mean() * 100:.1f}%")
    k3.metric("Median Salary", f"${df['base_salary_usd'].median():,.0f}")
    k4.metric("Median Tenure", f"{df['tenure_yrs'].median():.1f} yrs")

    st.markdown("### Attrition Trend (Last 12 Months)")
    st.line_chart(trend.set_index("month")["attrition_rate"])

    st.markdown("### Compensation Distribution")
    if view == "Overview":
        bins = pd.cut(df["base_salary_usd"], bins=10)
        comp = bins.value_counts().sort_index()
        comp_df = pd.DataFrame({"Salary Band (USD)": comp.index.astype(str), "Employees": comp.values})
        st.bar_chart(comp_df.set_index("Salary Band (USD)"))
    elif view == "By Level":
        by = df.groupby("level")["base_salary_usd"].median().sort_values(ascending=True)
        st.bar_chart(by)
    else:
        by = df.groupby("region")["base_salary_usd"].median().sort_values(ascending=True)
        st.bar_chart(by)

    with st.expander("Show synthetic dataset (for transparency)"):
        st.dataframe(df.head(50))
