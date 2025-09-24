# app.py
from __future__ import annotations
import importlib
import traceback
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import io
import base64
import requests

import pandas as pd
import streamlit as st

# PPTX helpers (self-contained; no extra files required)
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

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

# ---------- Improved PPTX snapshot builder (two-column, real tables) ----------
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import io
from datetime import datetime
import base64

def _load_logo_bytes() -> bytes | None:
    """
    Optional: add a small logo to top-right.
    Put a base64 png into secrets: 
      [branding]
      logo_b64 = "<base64>"
    Leave unset if you don't want a logo.
    """
    b64 = st.secrets.get("branding", {}).get("logo_b64")
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None

def _ppt_add_title(slide, title: str, subtitle: str | None, logo_bytes: bytes | None = None):
    # Title
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(8.5), Inches(0.9)).text_frame
    tb.text = title
    p = tb.paragraphs[0]; p.font.size = Pt(26); p.font.bold = True; p.font.color.rgb = RGBColor(11, 61, 145)

    # Subtitle + timestamp
    sb = slide.shapes.add_textbox(Inches(0.5), Inches(1.05), Inches(8.5), Inches(0.5)).text_frame
    stamp = datetime.now().strftime("%d %b %Y %H:%M")
    sb.text = (subtitle or "Auto summary") + f"  •  {stamp}"
    sp = sb.paragraphs[0]; sp.font.size = Pt(12); sp.font.color.rgb = RGBColor(90,90,90)

    # Optional logo
    if logo_bytes:
        bio = io.BytesIO(logo_bytes); bio.seek(0)
        slide.shapes.add_picture(bio, Inches(9.2), Inches(0.25), height=Inches(0.7))

def _add_caption(slide, text: str, left_in, top_in, width_in):
    cap = slide.shapes.add_textbox(Inches(left_in), Inches(top_in), Inches(width_in), Inches(0.3)).text_frame
    cap.text = text
    cap.paragraphs[0].font.size = Pt(11)
    cap.paragraphs[0].font.color.rgb = RGBColor(90, 90, 90)

def _ppt_add_fig(slide, fig, left_in, top_in, width_in, caption=""):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left_in), Inches(top_in), width=Inches(width_in))
    if caption:
        _add_caption(slide, caption, left_in, top_in + 2.85, width_in)
    return top_in + 3.15

def _df_to_table(slide, df: pd.DataFrame, left_in, top_in, width_in, max_rows=8, caption=""):
    """Render a compact, readable PPT table from a DataFrame (first max_rows)."""
    df = df.head(max_rows)
    rows, cols = len(df.index) + 1, len(df.columns)

    shape = slide.shapes.add_table(rows, cols, Inches(left_in), Inches(top_in), Inches(width_in), Inches(1.0))
    table = shape.table

    # Header
    for j, col in enumerate(df.columns):
        cell = table.cell(0, j)
        cell.text = str(col)
        run = cell.text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(11,61,145)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(240, 245, 255)

    # Body
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.text = "" if pd.isna(val) else str(val)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(9)
        # zebra banding
        if i % 2 == 0:
            for j in range(cols):
                c = table.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = RGBColor(250,250,250)

    # Column widths: simple even split
    for j in range(cols):
        table.columns[j].width = Inches(width_in / cols)

    # Caption (above)
    if caption:
        _add_caption(slide, caption, left_in, top_in - 0.28, width_in)

    return top_in + 1.25

def _build_one_pager(title: str, subtitle: str | None,
                     figs: list[tuple[str, "matplotlib.figure.Figure"]] | None = None,
                     tables: list[tuple[str, pd.DataFrame]] | None = None,
                     two_column: bool = True) -> bytes:
    """
    Produces a balanced one-pager:
      • Two columns by default
      • Crisp figures
      • Real PowerPoint tables (headers, banding)
    """
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    _ppt_add_title(slide, title, subtitle, logo_bytes=_load_logo_bytes())

    # Layout grid
    left_col, right_col = 0.5, 5.2
    col_w = 4.4
    y_left = y_right = 1.6

    def place_block(which: str, caption: str, payload):
        nonlocal y_left, y_right
        # pick column with lower Y
        left = (y_left <= y_right) or not two_column
        x = left_col if left or not two_column else right_col
        y = y_left if left or not two_column else y_right

        if which == "fig":
            new_y = _ppt_add_fig(slide, payload, x, y, col_w, caption=caption)
        else:
            new_y = _df_to_table(slide, payload, x, y + 0.25, col_w, caption=caption)

        if two_column:
            if left:
                y_left = new_y + 0.15
            else:
                y_right = new_y + 0.15
        else:
            y_left = new_y + 0.15

    for cap, fig in (figs or []):
        place_block("fig", cap, fig)
    for cap, df in (tables or []):
        place_block("tbl", cap, df)

    out = io.BytesIO(); prs.save(out); out.seek(0)
    return out.read()

# ---------- NEW: Graph-based mail sender (client credentials) ----------
def _get_graph_token(tenant_id: str, client_id: str, client_secret: str) -> str:
    """
    Request an access token using client credentials (app-only).
    Returns access_token (string) or raises RuntimeError.
    """
    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    payload = {
        "client_id": client_id,
        "scope": "https://graph.microsoft.com/.default",
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }
    resp = requests.post(token_url, data=payload, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Token request failed: {resp.status_code} {resp.text}")
    data = resp.json()
    return data.get("access_token")

def _send_via_graph(to_addrs: List[str], subject: str, html_body: str,
                    attachment_name: str, attachment_bytes: bytes):
    """
    Send mail as the mailbox user defined in secrets.toml [graph].mailbox
    using Microsoft Graph app-only sendMail (requires Mail.Send application permission + admin consent).
    """
    cfg = st.secrets.get("graph", {})
    tenant_id = cfg.get("tenant_id")
    client_id = cfg.get("client_id")
    client_secret = cfg.get("client_secret")
    mailbox = cfg.get("mailbox")

    if not all([tenant_id, client_id, client_secret, mailbox]):
        raise RuntimeError("Missing Graph configuration in secrets.toml under [graph].")

    token = _get_graph_token(tenant_id, client_id, client_secret)
    url = f"https://graph.microsoft.com/v1.0/users/{mailbox}/sendMail"

    # Build attachments per Graph fileAttachment schema (base64-encoded bytes)
    attachment_b64 = base64.b64encode(attachment_bytes).decode("ascii")
    attachments = [
        {
            "@odata.type": "#microsoft.graph.fileAttachment",
            "name": attachment_name,
            "contentType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "contentBytes": attachment_b64,
        }
    ]

    message = {
        "message": {
            "subject": subject,
            "body": {
                "contentType": "HTML",
                "content": html_body,
            },
            "toRecipients": [{"emailAddress": {"address": addr}} for addr in to_addrs],
            "attachments": attachments,
        },
        "saveToSentItems": "true",
    }

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    resp = requests.post(url, json=message, headers=headers, timeout=15)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Graph sendMail failed: {resp.status_code} {resp.text}")
    return True

# ============================== Share / Email snapshot UI ===============================
with st.expander("✉️  Share / Email snapshot"):
    # Pre-populated recipients (dropdown multi-select)
    default_recipients = st.secrets.get("graph", {}).get("recipients", []) or st.secrets.get("graph", {}).get("recipients", []) or st.secrets.get("email", {}).get("recipients", [])
    selected = st.multiselect("Choose recipient(s)", options=default_recipients, default=default_recipients[:1] if default_recipients else [])
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

                # Send using Graph
                _send_via_graph(
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
