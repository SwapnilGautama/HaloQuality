# app.py
from __future__ import annotations
import importlib
import traceback
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import io
import smtplib

import pandas as pd
import streamlit as st

# PPTX + email helpers (self-contained; no extra files required)
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

# =============== Small import helper (keeps Q1 & Q2 isolated) ===============
def _imp(mod: str, attr: str | None = None):
    """Import from repo root; if that fails, from core.<mod>."""
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError:
        m = importlib.import_module(f"core.{mod}")
    return getattr(m, attr) if attr else m

# These must exist in your repo (root or core/)
load_store = _imp("data_store", "load_store")
sem_router = _imp("semantic_router")  # must define match(q) -> {"slug": ..., "params": {...}}

# Question modules are always looked up here (keeps them sandboxed from each other)
QUESTION_MODULE_PREFIXES = ("questions", "core.questions")

def _run_question(store: Dict[str, Any], slug: str, params: Dict[str, Any], user_text: Optional[str] = None):
    """
    Dynamically import a question module and run it.
    Every question exposes: run(store, params, user_text=None) -> (message|tuple, optional_df)
    """
    last_exc = None
    for prefix in QUESTION_MODULE_PREFIXES:
        try:
            mod = importlib.import_module(f"{prefix}.{slug}")
            return mod.run(store, params, user_text=user_text)
        except Exception as e:
            last_exc = e
            continue

    err = f"That question module failed to import.\n\nslug={slug}\n\n{traceback.format_exc()}"
    return err, pd.DataFrame()

# ===================== Page setup ====================
st.set_page_config(page_title="Halo - Quality - AI Assistant", layout="wide")

# ---------- Header styles (no sidebar rule here) ----------
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] { display:none !important; }

      /* HALO branding */
      .halo-wrap{
        display:flex; align-items:baseline; gap:.75rem; margin:8px 0 22px 0;
      }
      .halo-pill{
        background: linear-gradient(90deg, #FF7A00 0%, #FFD54F 50%, #66BB6A 100%);
        color:white; font-weight:900; letter-spacing:1.2px;
        border-radius:12px; padding:8px 16px; display:inline-block; font-size:28px;
        text-shadow: 0 1px 1px rgba(0,0,0,.18);
      }
      .brand-title{
        color:#0B3B8C; font-weight:800; font-size:36px; line-height:1.1;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
    <div class="halo-wrap">
      <div class="halo-pill">HALO</div>
      <div class="brand-title">- Quality - AI Assistant</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================== Load “store” ==============================
with st.spinner("Loading data..."):
    try:
        store = load_store(assume_year_for_complaints=2025)
    except TypeError:
        store = load_store()

# ============================== Router + query box ==============================
# FIX: initialize once; let the widget own the value thereafter
if "q" not in st.session_state:
    st.session_state["q"] = "fpa"   # one-time default on first load only

q = st.text_input(
    "Type your question (e.g., 'comp', 'complaint', 'nps' or 'fpa')",
    key="q",   # <- binds state; no 'value=' so it won't be reset on reruns
)

# Route
user_query = (q or "").strip()
match = sem_router.match(user_query) if hasattr(sem_router, "match") else {"slug": "complaints_june_by_portfolio", "params": {}}
slug = match.get("slug", "complaints_june_by_portfolio")
params = match.get("params", {}) or {}

# ============================== Conditional sidebar visibility ===============================
def _wants_sidebar(text: str, p: Dict[str, Any]) -> bool:
    if p.get("show_sidebar") is True:
        return True
    t = (text or "").lower()
    keywords = ("filter", "filters", "filter pane", "with filters", "show filters")
    return any(k in t for k in keywords)

SHOW_SIDEBAR = _wants_sidebar(user_query, params)

# Apply CSS to hide sidebar only when NOT requested
if not SHOW_SIDEBAR:
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"], [data-testid="stSidebarNav"] { display: none !important; }
          section[data-testid="stSidebar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ============================== EMAIL SNAPSHOT HELPERS ===============================
def _import_question_module(slug: str):
    last_exc = None
    for prefix in QUESTION_MODULE_PREFIXES:
        try:
            return importlib.import_module(f"{prefix}.{slug}")
        except Exception as e:
            last_exc = e
            continue
    return None

def _ppt_add_title(slide, title: str, subtitle: str | None):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.9)).text_frame
    tb.text = title
    p = tb.paragraphs[0]
    p.font.size = Pt(26)
    p.font.bold = True
    p.font.color.rgb = RGBColor(11, 61, 145)

    sb = slide.shapes.add_textbox(Inches(0.5), Inches(1.05), Inches(9), Inches(0.5)).text_frame
    stamp = datetime.now().strftime("%d %b %Y %H:%M")
    sb.text = (subtitle or "").strip() + (("  •  " + stamp) if subtitle else stamp)
    sb.paragraphs[0].font.size = Pt(12)
    sb.paragraphs[0].font.color.rgb = RGBColor(90, 90, 90)

def _ppt_add_fig(slide, fig, caption: str, x: float, y: float, w: float) -> float:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(x), Inches(y), width=Inches(w))
    cap = slide.shapes.add_textbox(Inches(x), Inches(y + 2.9), Inches(w), Inches(0.3)).text_frame
    cap.text = caption
    cap.paragraphs[0].font.size = Pt(10)
    cap.paragraphs[0].font.color.rgb = RGBColor(90, 90, 90)
    return y + 3.2

def _ppt_add_table_text(slide, df: pd.DataFrame, caption: str, x: float, y: float, w: float) -> float:
    txt = df.head(10).to_string(index=False)
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(1.6)).text_frame
    box.text = caption
    box.paragraphs[0].font.size = Pt(12)
    box.paragraphs[0].font.bold = True
    p = box.add_paragraph()
    p.text = txt
    p.font.size = Pt(10)
    return y + 1.8

def _build_one_pager(title: str, subtitle: str | None,
                     figs: List[Tuple[str, "matplotlib.figure.Figure"]] | None = None,
                     tables: List[Tuple[str, pd.DataFrame]] | None = None) -> bytes:
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _ppt_add_title(slide, title, subtitle)
    x, y, w = 0.5, 1.6, 9.2
    if figs:
        for cap, fig in figs:
            y = _ppt_add_fig(slide, fig, cap, x, y, w)
    if tables:
        for cap, df in tables:
            y = _ppt_add_table_text(slide, df, cap, x, y, w)
    out = io.BytesIO()
    prs.save(out)
    out.seek(0)
    return out.read()

def _get_snapshot_content(slug: str, store: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the question module exposes build_snapshot(store, params) -> dict
    with keys: title, subtitle, figs=[(cap, fig)], tables=[(cap, df)],
    we use it. Otherwise, create a simple, useful fallback.
    """
    mod = _import_question_module(slug)
    if mod and hasattr(mod, "build_snapshot"):
        try:
            snap = mod.build_snapshot(store, params)
            if isinstance(snap, dict):
                return snap
        except Exception:
            pass

    # Fallback (generic preview from store)
    title = f"Halo Quality — {slug.replace('_',' ').title()} Snapshot"
    subtitle = "Auto summary"
    tables: List[Tuple[str, pd.DataFrame]] = []
    for key in ("fpa", "nps", "complaints"):
        obj = store.get(key)
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            tables.append((f"{key.upper()} (preview)", obj.head(8)))
    if not tables:
        tables.append(("Info", pd.DataFrame({"note": ["No snapshot function; add build_snapshot() to module."]})))
    return {"title": title, "subtitle": subtitle, "figs": [], "tables": tables}

def _send_via_outlook(to_addrs: List[str], subject: str, html_body: str,
                      attachment_name: str, attachment_bytes: bytes):
    cfg = st.secrets.get("email", {})
    smtp_host = cfg.get("smtp_host", "smtp.office365.com")
    smtp_port = int(cfg.get("smtp_port", 587))
    username  = cfg.get("username")
    password  = cfg.get("password")
    sender    = cfg.get("sender", username)

    if not all([smtp_host, smtp_port, username, password, sender]):
        raise RuntimeError("Email configuration missing in secrets.toml under [email].")

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html"))

    part = MIMEBase("application", "vnd.openxmlformats-officedocument.presentationml.presentation")
    part.set_payload(attachment_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
    msg.attach(part)

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(sender, to_addrs, msg.as_string())

# ============================== Share / Email snapshot UI ===============================
with st.expander("✉️  Share / Email snapshot"):
    # Pre-populated recipients (dropdown multi-select)
    default_recipients = st.secrets.get("email", {}).get("recipients", [])
    selected = st.multiselect("Choose recipient(s)", options=default_recipients, default=default_recipients[:1])
    custom = st.text_input("Or add another recipient (optional)", placeholder="name@company.com")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fmt = st.selectbox("Format", ["PPTX (one page)"], index=0)
    with col_b:
        include_tables = st.checkbox("Include small summary tables", value=True)

    if st.button("Send snapshot"):
        to_list = list(selected)
        if custom and "@" in custom:
            to_list.append(custom)

        if not to_list:
            st.error("Pick or enter at least one email address.")
        else:
            try:
                snap = _get_snapshot_content(slug, store, params)
                title = snap.get("title") or f"Halo Quality — {slug}"
                subtitle = snap.get("subtitle") or ""
                figs = snap.get("figs") or []
                tables = snap.get("tables") if include_tables else []
                ppt_bytes = _build_one_pager(title, subtitle, figs=figs, tables=tables)

                _send_via_outlook(
                    to_addrs=to_list,
                    subject=title,
                    html_body=f"<p>Hi,</p><p>Attached is the snapshot from <b>Halo Quality</b> ({slug}).</p>",
                    attachment_name=f"{slug}_snapshot.pptx",
                    attachment_bytes=ppt_bytes,
                )
                st.success(f"Snapshot sent to: {', '.join(to_list)} ✅")
            except Exception as e:
                st.error(f"Could not send email: {e}")

# ============================== Run question ===============================
try:
    result, df = _run_question(store, slug, params, user_text=user_query)
except Exception:
    st.error("Sorry—couldn't run that question.")
    st.code(traceback.format_exc())
else:
    if isinstance(result, tuple) and len(result) in (1, 2):
        title = result[0]
        subtitle = result[1] if len(result) == 2 else None
        if isinstance(title, str) and title.strip():
            st.subheader(title)
        if subtitle:
            st.caption(subtitle)
    elif isinstance(result, str) and result.strip():
        st.info(result)

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.dataframe(df, use_container_width=True)
