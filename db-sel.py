import os
import re

import streamlit as st
from sqlalchemy import inspect, text

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="🤖 Database Agent", page_icon="🎯", layout="centered")
st.title("🤖 Talk to DB")
st.markdown("**Gemini + PostgreSQL** – schema-aware SQL, safer + faster. ❄️")


# ----------------------------
# Session state init
# ----------------------------
if "active_db" not in st.session_state:
    st.session_state.active_db = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "preview_table" not in st.session_state:
    st.session_state.preview_table = None
if "preview_page" not in st.session_state:
    st.session_state.preview_page = 1


# ----------------------------
# Pricing / token estimate (approx)
# ----------------------------
USD_TO_INR = 90.0
USD_PER_1M_TOKENS = 0.50  # approx for gemini-2.0-flash total (input+output)
INR_PER_1M_TOKENS = USD_PER_1M_TOKENS * USD_TO_INR  # = ₹45

def estimate_tokens(text: str) -> int:
    # ~1 token per 4 chars (rough)
    if not text:
        return 0
    return max(1, len(text) // 4)

def estimate_cost_inr(tokens: int) -> float:
    return (tokens / 1_000_000.0) * INR_PER_1M_TOKENS


# ----------------------------
# Helpers
# ----------------------------
RESERVED = {"user", "order"}

def quote_ident(name: str) -> str:
    if not name:
        return name
    n = str(name)
    if (not n.islower()) or (" " in n) or (n.lower() in RESERVED):
        return f'"{n.replace(chr(34), chr(34) * 2)}"'
    return n

def extract_sql(text_: str) -> str:
    t = (text_ or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0].strip()
    return t.strip()

def has_multiple_statements(sql: str) -> bool:
    s = (sql or "").strip()
    return s.count(";") > 1

# ✅ safer keyword blocking: match whole SQL keywords only
BLOCKED_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "copy", "call", "execute"
]
BLOCKED_RE = re.compile(r"(?is)\b(" + "|".join(BLOCKED_KEYWORDS) + r")\b")

def is_safe_select(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False

    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False

    # allow a single trailing semicolon
    if has_multiple_statements(s):
        return False

    # ✅ only block real keywords (won't block 'updated_timestamp')
    if BLOCKED_RE.search(low):
        return False

    return True

MAX_LIMIT = 200

def enforce_limit(sql: str, max_limit: int = MAX_LIMIT) -> str:
    s = (sql or "").strip().rstrip(";")
    if re.search(r"(?i)\blimit\b", s):
        s = re.sub(
            r"(?i)\blimit\s+(\d+)",
            lambda m: f"LIMIT {min(int(m.group(1)), max_limit)}",
            s
        )
        return s + ";"
    return s + f"\nLIMIT {max_limit};"


# ----------------------------
# ENUM helpers (Option A)
# ----------------------------
@st.cache_data(ttl=600)
def get_enum_labels(db_key: str, _engine, enum_type_name: str) -> list[str]:
    """
    Fetch enum labels from Postgres system catalogs by enum type name.
    Works even if SQLAlchemy doesn't expose .enums.
    """
    if not enum_type_name:
        return []

    q = text("""
        SELECT e.enumlabel
        FROM pg_type t
        JOIN pg_enum e ON t.oid = e.enumtypid
        JOIN pg_namespace n ON n.oid = t.typnamespace
        WHERE t.typname = :typname
        ORDER BY e.enumsortorder;
    """)

    try:
        with _engine.connect() as cxn:
            rows = cxn.execute(q, {"typname": enum_type_name}).fetchall()
        return [r[0] for r in rows] if rows else []
    except Exception:
        return []

def _fmt_col_for_llm(db_key: str, _engine, col: dict) -> str:
    """
    Format a column as: name TYPE or name ENUM('A','B',...)
    Prefer SQLAlchemy .enums; fallback to pg catalogs using type.name.
    """
    col_name = col["name"]
    col_type = col["type"]

    # 1) Best case: SQLAlchemy enum exposes allowed values
    enum_vals = getattr(col_type, "enums", None)
    if enum_vals:
        vals = ", ".join(repr(v) for v in enum_vals)
        return f"{col_name} ENUM({vals})"

    # 2) Fallback: try to detect enum type name and query pg catalogs
    type_name = getattr(col_type, "name", None)  # e.g. 'driver_value_type' etc.
    if type_name:
        labels = get_enum_labels(db_key, _engine, type_name)
        if labels:
            vals = ", ".join(repr(v) for v in labels)
            return f"{col_name} ENUM({vals})"

    # 3) Default: include basic type to help the model (kept compact)
    return f"{col_name} {str(col_type)}"


# ----------------------------
# Schema fetch (tables + compact schema for LLM) — cached PER DB
# ----------------------------
@st.cache_data(ttl=600)
def get_tables_sorted(db_key: str, _engine) -> list[str]:
    insp = inspect(_engine)
    tables = insp.get_table_names()
    return sorted(tables, key=lambda x: x.lower())

@st.cache_data(ttl=600)
def get_compact_schema_for_llm(db_key: str, _engine) -> str:
    insp = inspect(_engine)
    tables = sorted(insp.get_table_names(), key=lambda x: x.lower())
    lines = []
    for t in tables:
        cols = insp.get_columns(t)
        parts = [_fmt_col_for_llm(db_key, _engine, c) for c in cols]
        lines.append(f"{t}({', '.join(parts)})")
    return "\n".join(lines)

@st.cache_data(ttl=600)
def get_schema_for_tables(db_key: str, _engine, table_subset: tuple[str, ...]) -> str:
    """Return schema only for selected tables (huge token saver), including ENUM values."""
    insp = inspect(_engine)
    lines = []
    for t in table_subset:
        cols = insp.get_columns(t)
        parts = [_fmt_col_for_llm(db_key, _engine, c) for c in cols]
        lines.append(f"{t}({', '.join(parts)})")
    return "\n".join(lines)

def pick_relevant_tables(prompt: str, tables: list[str], max_tables: int = 3) -> list[str]:
    """Heuristic: pick tables whose names match prompt words (users->user)."""
    q = (prompt or "").lower()
    words = re.findall(r"[a-zA-Z_]+", q)

    scored = []
    for t in tables:
        tl = t.lower()
        score = 0
        if tl in q:
            score += 5
        for w in words:
            w0 = w.rstrip("s")
            t0 = tl.rstrip("s")
            if w == tl:
                score += 4
            if w0 == t0:
                score += 3
            if w in tl or tl in w:
                score += 1
        if score > 0:
            scored.append((score, t))

    scored.sort(reverse=True)
    return [t for _, t in scored[:max_tables]]


# ----------------------------
# Gemini LLM — cached resource (keep output short)
# ----------------------------
@st.cache_resource
def get_llm():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    model = st.secrets.get("GEMINI_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        google_api_key=api_key,
        max_output_tokens=200,  # ✅ smaller output => fewer tokens
    )

try:
    llm = get_llm()
except Exception as e:
    st.error(f"❌ Gemini init error: {e}")
    st.stop()


# ----------------------------
# SQL generation chain (short prompt)
# ----------------------------
sql_prompt = PromptTemplate.from_template("""
You are a PostgreSQL expert. Output ONE SELECT query only.

Use ONLY the schema below. Do NOT guess columns.
Prefer simple queries. Avoid joins unless asked.
If user asks "latest/newest/recent", order by id DESC (if id exists).

Schema:
{schema}

Rules:
- Only SELECT or WITH.
- Always include LIMIT <= {max_limit}.
- For ENUM columns, use one of the provided ENUM(...) values exactly (case-sensitive).
- If filtering on text columns (name, email, title, etc.) and the value looks partial or user did not say "exact",
  use ILIKE with %value%.
- Return ONLY SQL.

Question: {question}
SQL:
""")


generate_sql = sql_prompt | llm | StrOutputParser()

answer_prompt = PromptTemplate.from_template("""
Reply in Hinglish in maximum 2-3 short lines.
No long explanation.

Question: {question}
Result rows (sample): {response}

Answer:
""")
answer_chain = answer_prompt | llm | StrOutputParser()


# ----------------------------
# Sidebar: DB select + Table select
# ----------------------------
with st.sidebar:
    st.header("🗄️ Database")

    db_choice = st.selectbox(
        "Select DB",
        options=[
            ("Volume DB", "postgresql_volume"),
            ("Quorbit DB", "postgresql_quorbit"),
        ],
        format_func=lambda x: x[0],
        index=0,
        key="db_choice",
    )

    db_label, db_conn_name = db_choice

    if st.session_state.active_db != db_conn_name:
        st.session_state.active_db = db_conn_name
        st.session_state.messages = []
        st.session_state.preview_table = None
        st.session_state.preview_page = 1
        st.session_state.pop("selected_table", None)

        # reset preview control widgets (optional)
        st.session_state.pop("page_size", None)
        st.session_state.pop("sort_col", None)
        st.session_state.pop("sort_dir", None)
        st.session_state.pop("page", None)

        st.toast(f"Switched to {db_label}", icon="🔄")

    st.divider()


# ----------------------------
# PostgreSQL connection
# ----------------------------
conn = st.connection(st.session_state.active_db, type="sql")

try:
    conn.query("SELECT 1", ttl=0)
    st.toast(f"Connected to {db_label}", icon="✅")
except Exception as e:
    st.error(f"❌ Connection failed: {e}")
    st.stop()


table_names = get_tables_sorted(st.session_state.active_db, conn.engine)
schema_full_compact = get_compact_schema_for_llm(st.session_state.active_db, conn.engine)

with st.sidebar:
    st.header("📊 Tables")

    selected_table = st.selectbox(
        "Select table",
        options=table_names,
        index=None,
        placeholder="Select a table…",
        key="selected_table",
    )

    st.divider()
    if st.button("Clear screen", width="stretch"):
        st.session_state.messages = []
        st.session_state.preview_table = None
        st.session_state.preview_page = 1
        st.session_state.pop("selected_table", None)
        st.toast("🧼 Cleared!", icon="🧼")
        st.rerun()


# ----------------------------
# Main: Table Preview (default: latest first by id DESC if id exists)
# ----------------------------
st.markdown("**📋 Table Preview**")

if selected_table is None:
    st.info("👈 Select a table from the left to preview data.")
else:
    safe_table = quote_ident(selected_table)

    cols_df = conn.query(f"SELECT * FROM {safe_table} LIMIT 0;", ttl=600)
    table_cols = list(cols_df.columns)

    if st.session_state.preview_table != selected_table:
        st.session_state.preview_table = selected_table
        st.session_state.preview_page = 1

    with st.expander("⚙️ Preview settings", expanded=False):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            page_size = st.number_input("Page size", 10, 500, 100, step=10, key="page_size")
        with c2:
            sort_col = st.selectbox("Sort column", ["(none)"] + table_cols, index=0, key="sort_col")
        with c3:
            sort_dir = st.radio("Direction", ["ASC", "DESC"], horizontal=True, index=1, key="sort_dir")  # default DESC
        with c4:
            page = st.number_input("Page", 1, value=int(st.session_state.preview_page), step=1, key="page")

        st.session_state.preview_page = int(page)

        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if st.button("⬅️ Prev", width="stretch", disabled=(page <= 1), key="prev_btn"):
                st.session_state.preview_page = max(1, page - 1)
                st.rerun()
        with nav2:
            if st.button("Next ➡️", width="stretch", key="next_btn"):
                st.session_state.preview_page = page + 1
                st.rerun()

    offset = (st.session_state.preview_page - 1) * int(page_size)

    order_clause = ""
    if st.session_state.get("sort_col") and st.session_state.sort_col != "(none)":
        order_clause = f"ORDER BY {quote_ident(st.session_state.sort_col)} {st.session_state.sort_dir}"
    else:
        lower_cols = [c.lower() for c in table_cols]
        if "id" in lower_cols:
            id_col = table_cols[lower_cols.index("id")]
            order_clause = f"ORDER BY {quote_ident(id_col)} DESC"

    preview_sql = f"""
    SELECT *
    FROM {safe_table}
    {order_clause}
    LIMIT {int(page_size)} OFFSET {int(offset)};
    """.strip()

    df_preview = conn.query(preview_sql, ttl=0)
    st.caption(f"**{selected_table}** — Rows {offset + 1} to {offset + len(df_preview)}")
    st.dataframe(df_preview, width="stretch", height=420)


# ----------------------------
# Chat history render
# ----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            st.caption("🧾 SQL:")
            st.code(message["sql"], language="sql")


# ----------------------------
# Chat input + execution + token/cost display
# ----------------------------
if prompt := st.chat_input("Ask DB..."):
    # show user message now
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # ✅ token saver: if a table is selected, prefer that schema only.
            candidate_tables = [selected_table] if selected_table else pick_relevant_tables(prompt, table_names, max_tables=3)

            if candidate_tables:
                schema_small = get_schema_for_tables(
                    st.session_state.active_db, conn.engine, tuple(candidate_tables)
                )
            else:
                schema_small = schema_full_compact

            sql_raw = generate_sql.invoke({"question": prompt, "schema": schema_small, "max_limit": str(MAX_LIMIT)})
            sql_query = extract_sql(sql_raw)

            if not is_safe_select(sql_query):
                raise ValueError(f"Unsafe SQL:\n{sql_query}")

            sql_query = enforce_limit(sql_query, MAX_LIMIT)

            st.caption("🧾 SQL")
            st.code(sql_query, language="sql")

            df = conn.query(sql_query, ttl=0)
            st.dataframe(df, width="stretch")

            # small sample to reduce tokens
            raw_rows = df.head(10).to_dict(orient="records")

            response = answer_chain.invoke({"question": prompt, "response": raw_rows})
            st.markdown(response)

            # --- token & price (estimated) ---
            prompt_tokens = estimate_tokens(prompt)
            schema_tokens = estimate_tokens(schema_small)
            sql_tokens = estimate_tokens(sql_query)
            result_tokens = estimate_tokens(str(raw_rows))
            answer_tokens = estimate_tokens(response)
            total_tokens = prompt_tokens + schema_tokens + sql_tokens + result_tokens + answer_tokens

            cost_total = estimate_cost_inr(total_tokens)
            cost_schema = estimate_cost_inr(schema_tokens)
            cost_result = estimate_cost_inr(result_tokens)

            with st.expander("📊 Token & Cost (estimated)", expanded=False):
                st.write(f"🧠 Prompt: ~{prompt_tokens} tokens (₹{estimate_cost_inr(prompt_tokens):.4f})")
                st.write(f"🗂️ Schema: ~{schema_tokens} tokens (₹{cost_schema:.4f})")
                st.write(f"🧾 SQL: ~{sql_tokens} tokens (₹{estimate_cost_inr(sql_tokens):.4f})")
                st.write(f"📋 Result sample: ~{result_tokens} tokens (₹{cost_result:.4f})")
                st.write(f"💬 Answer: ~{answer_tokens} tokens (₹{estimate_cost_inr(answer_tokens):.4f})")
                st.markdown(f"### ✅ Total: ~**{total_tokens} tokens** → **₹{cost_total:.4f}**")
                if candidate_tables:
                    st.caption(f"Schema reduced ✅ tables used: {', '.join(candidate_tables)}")
                st.caption(f"USD→INR=90, approx pricing used: $0.50 / 1M tokens (~₹45 / 1M)")

            st.session_state.messages.append({"role": "assistant", "content": response, "sql": sql_query})

        except Exception as e:
            response = f"Bhai error aa gaya 😅: {e}"
            st.error(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


st.markdown("---")
st.caption("Made with ❤️ | Gemini + PostgreSQL + Streamlit")
