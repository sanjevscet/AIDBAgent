import os
import re
import pandas as pd

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
if "write_logs" not in st.session_state:
    st.session_state.write_logs = []  # {"sql":..., "params":...}
if "open_custom_sql" not in st.session_state:
    st.session_state.open_custom_sql = False


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

# READ-only blocked keywords (LLM-generated SQL)
BLOCKED_KEYWORDS_READ = [
    "insert", "update", "delete", "drop", "alter", "truncate",
    "create", "grant", "revoke", "copy", "call", "execute"
]
BLOCKED_RE_READ = re.compile(r"(?is)\b(" + "|".join(BLOCKED_KEYWORDS_READ) + r")\b")

def is_safe_select(sql: str) -> bool:
    s = (sql or "").strip()
    if not s:
        return False
    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False
    if has_multiple_statements(s):
        return False
    if BLOCKED_RE_READ.search(low):
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

def to_py(v):
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return v


# ----------------------------
# Table name normalizer (plural -> singular)
# ----------------------------
def normalize_table_names(sql: str, actual_tables: list[str]) -> str:
    if not sql:
        return sql

    tables_lower = {t.lower(): t for t in actual_tables}

    def fix_table(token: str) -> str:
        t = token.strip()
        tl = t.lower()
        if tl in tables_lower:
            return tables_lower[tl]
        if tl.endswith("s"):
            singular = tl[:-1]
            if singular in tables_lower:
                return tables_lower[singular]
        return t

    sql = re.sub(
        r"(?is)\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        lambda m: f"FROM {fix_table(m.group(1))}",
        sql
    )
    sql = re.sub(
        r"(?is)\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        lambda m: f"JOIN {fix_table(m.group(1))}",
        sql
    )
    return sql


# ----------------------------
# ✅ Fix common FK join guesses (driver.driver_id -> driver.id)
# ----------------------------
def fix_bad_fk_join(sql: str, table_names: list[str], table_cols_map: dict[str, list[str]]) -> str:
    """
    Fix common FK join mistakes like:
      driver d JOIN driver_category_assignment dca ON d.driver_id = dca.driver_id
    when driver table doesn't have driver_id but has id.

    Also fixes SELECT DISTINCT d.driver_id -> d.id if driver_id doesn't exist.
    """
    if not sql:
        return sql

    s = sql

    # Parse base table + alias from FROM
    m_from = re.search(r"(?is)\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+([a-zA-Z_][a-zA-Z0-9_]*))?", s)
    if not m_from:
        return s

    base_tbl = m_from.group(1)
    base_alias = m_from.group(2) or base_tbl

    # normalize base table
    actual_map = {t.lower(): t for t in table_names}
    base_tbl_actual = actual_map.get(base_tbl.lower(), base_tbl)

    base_cols = [c.lower() for c in table_cols_map.get(base_tbl_actual, [])]

    # If driver_id not in base table but id exists, fix SELECT alias.driver_id -> alias.id
    if "driver_id" not in base_cols and "id" in base_cols:
        s = re.sub(
            rf"(?is)\b{re.escape(base_alias)}\.driver_id\b",
            f"{base_alias}.id",
            s
        )

        # Fix JOIN pattern: alias.driver_id = other.driver_id -> alias.id = other.driver_id
        s = re.sub(
            rf"(?is)\b{re.escape(base_alias)}\.driver_id\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*\.)?driver_id\b",
            f"{base_alias}.id = \\1driver_id",
            s
        )

    return s


# ----------------------------
# ✅ "latest/newest/recent" ORDER BY fixer (prefer id DESC) + JOIN-safe
# ----------------------------
def prefer_id_order_for_latest(sql: str, table_names: list[str], table_to_cols: dict[str, list[str]]) -> str:
    """
    If ORDER BY uses created_at/updated_at OR plain 'id' in a joined query,
    prefer ORDER BY <base_alias>.id DESC (if id exists on base table).
    Also fixes ambiguous ORDER BY id when JOIN exists.
    """
    if not sql:
        return sql

    s = sql

    # FROM driver d  -> base_tbl=driver, base_alias=d (alias optional)
    m = re.search(r"(?is)\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+([a-zA-Z_][a-zA-Z0-9_]*))?", s)
    if not m:
        return s

    base_tbl = m.group(1)
    base_alias = m.group(2)  # may be None

    actual_map = {t.lower(): t for t in table_names}
    actual_tbl = actual_map.get(base_tbl.lower(), base_tbl)

    cols = table_to_cols.get(actual_tbl, [])
    cols_l = [c.lower() for c in cols]
    if "id" not in cols_l:
        return s

    id_col = cols[cols_l.index("id")]
    id_expr = f"{base_alias}.{quote_ident(id_col)}" if base_alias else f"{quote_ident(id_col)}"

    has_join = re.search(r"(?is)\bjoin\b", s) is not None

    # (A) Fix ambiguous ORDER BY id in join queries
    if has_join or base_alias:
        s = re.sub(
            r"(?is)\border\s+by\s+id(\s+(asc|desc))?\b",
            lambda _mm: f"ORDER BY {id_expr} DESC",
            s
        )

    # (B) Replace created_at/updated_at ordering -> id ordering (qualified)
    s = re.sub(
        r"(?is)\border\s+by\s+(created_at|updated_at|createdon|updatedon|created|updated|timestamp|time|date)(\s+(asc|desc))?\b",
        lambda _mm: f"ORDER BY {id_expr} DESC",
        s
    )

    # (C) If no ORDER BY but has LIMIT, add ORDER BY <alias>.id DESC
    if (re.search(r"(?is)\blimit\b", s) is not None) and (re.search(r"(?is)\border\s+by\b", s) is None):
        s = re.sub(r"(?is)\blimit\b", f"ORDER BY {id_expr} DESC\nLIMIT", s, count=1)

    return s


# ----------------------------
# Editable table helpers (PK based update/delete)
# ----------------------------
def get_primary_key_cols(_engine, table_name: str) -> list[str]:
    insp = inspect(_engine)
    try:
        pk = insp.get_pk_constraint(table_name) or {}
        return pk.get("constrained_columns") or []
    except Exception:
        return []

def build_update_sql(table_sql: str, set_cols: list[str], pk_cols: list[str]) -> str:
    set_clause = ", ".join([f"{quote_ident(c)} = :set_{c}" for c in set_cols])
    where_clause = " AND ".join([f"{quote_ident(c)} = :pk_{c}" for c in pk_cols])
    return f"UPDATE {table_sql} SET {set_clause} WHERE {where_clause} RETURNING *;"

def build_delete_sql(table_sql: str, pk_cols: list[str]) -> str:
    where_clause = " AND ".join([f"{quote_ident(c)} = :pk_{c}" for c in pk_cols])
    return f"DELETE FROM {table_sql} WHERE {where_clause} RETURNING *;"

def _eq(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b

def diff_rows(original_df: pd.DataFrame, edited_df: pd.DataFrame, pk_cols: list[str], delete_col: str = "__delete__"):
    if original_df is None or edited_df is None or original_df.empty:
        return [], []

    orig = original_df.copy()
    edt = edited_df.copy()

    deletes = []
    if delete_col in edt.columns:
        del_rows = edt[edt[delete_col] == True]
        for _, r in del_rows.iterrows():
            deletes.append({c: r[c] for c in pk_cols})

    edt_cmp = edt.drop(columns=[delete_col]) if delete_col in edt.columns else edt

    def pk_key(df: pd.DataFrame) -> pd.Series:
        return df[pk_cols].astype(str).agg("||".join, axis=1)

    orig2 = orig.copy()
    edt2 = edt_cmp.copy()
    orig2["__pkkey__"] = pk_key(orig2).values
    edt2["__pkkey__"] = pk_key(edt2).values

    orig_map = orig2.set_index("__pkkey__")
    edt_map = edt2.set_index("__pkkey__")

    updates = []
    common_keys = orig_map.index.intersection(edt_map.index)
    edt_cmp_key = pk_key(edt_cmp)

    for k in common_keys:
        o = orig_map.loc[k]
        e = edt_map.loc[k]

        if delete_col in edt.columns:
            row_mask = edt_cmp_key == k
            if any(row_mask):
                row_flag = bool(edt.loc[row_mask, delete_col].iloc[0])
                if row_flag:
                    continue

        changed_cols = []
        for col in edt_cmp.columns:
            if col in pk_cols:
                continue
            if col not in orig.columns:
                continue
            if not _eq(o[col], e[col]):
                changed_cols.append(col)

        if changed_cols:
            payload = {}
            for c in pk_cols:
                payload[f"pk_{c}"] = e[c]
            for c in changed_cols:
                payload[f"set_{c}"] = e[c]
            payload["__set_cols__"] = changed_cols
            updates.append(payload)

    return updates, deletes


# ----------------------------
# Custom SQL runner (Allowlist: SELECT/INSERT/UPDATE/DELETE only)
# ----------------------------
FORBIDDEN_KEYWORDS = [
    "drop", "alter", "truncate", "create", "grant", "revoke",
    "copy", "call", "execute", "vacuum", "analyze", "refresh",
    "cluster", "reindex", "attach", "detach"
]
FORBIDDEN_RE = re.compile(r"(?is)\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b")

def strip_sql_comments(sql: str) -> str:
    s = re.sub(r"(?is)/\*.*?\*/", " ", sql or "")
    s = re.sub(r"(?m)--.*?$", " ", s)
    return s

def normalize_sql(sql: str) -> str:
    s = extract_sql(sql or "")
    s = strip_sql_comments(s)
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1].rstrip()
    return s

def classify_sql(sql: str) -> str:
    s = (sql or "").strip().lower()
    if s.startswith("with"):
        return "with"
    if s.startswith("select"):
        return "select"
    if s.startswith("insert"):
        return "insert"
    if s.startswith("update"):
        return "update"
    if s.startswith("delete"):
        return "delete"
    return "other"

def has_where_clause(sql: str) -> bool:
    return re.search(r"(?is)\bwhere\b", sql or "") is not None

def is_single_statement(sql: str) -> bool:
    return (sql or "").count(";") == 0

def validate_custom_sql(sql: str) -> tuple[bool, str, str]:
    s = normalize_sql(sql)
    if not s:
        return False, "other", "Empty SQL"
    if not is_single_statement(s):
        return False, "other", "Only single statement allowed (no multiple ';')"
    if FORBIDDEN_RE.search(s):
        return False, "other", "Forbidden keyword found (DDL/admin ops blocked)"

    kind = classify_sql(s)
    if kind == "other":
        return False, kind, "Only SELECT / INSERT / UPDATE / DELETE are allowed"

    if kind in ("update", "delete") and not has_where_clause(s):
        return False, kind, f"{kind.upper()} must include WHERE (full-table ops blocked)"

    return True, kind, "OK"

def enforce_limit_for_select_or_with(sql: str, max_limit: int = MAX_LIMIT) -> str:
    s = normalize_sql(sql)
    low = s.lower().lstrip()
    if low.startswith("select"):
        return enforce_limit(s, max_limit).strip().rstrip(";")
    if low.startswith("with"):
        if re.search(r"(?is)\bwith\b.*\bselect\b", s) and not re.search(r"(?is)\bwith\b.*\b(insert|update|delete)\b", s):
            return enforce_limit(s, max_limit).strip().rstrip(";")
    return s

def run_custom_sql(conn, sql: str):
    s = normalize_sql(sql)
    kind = classify_sql(s)

    if kind in ("select", "with"):
        s2 = enforce_limit_for_select_or_with(s, MAX_LIMIT)
        df = conn.query(s2, ttl=0)
        return {"kind": "select", "sql": s2, "df": df}

    with conn.engine.begin() as cxn:
        result = cxn.execute(text(s))
        returning_rows = None
        try:
            rows = result.fetchall()
            returning_rows = [dict(r._mapping) for r in rows]
        except Exception:
            returning_rows = None
        rowcount = getattr(result, "rowcount", None)

    df_ret = pd.DataFrame(returning_rows) if returning_rows else None
    return {"kind": kind, "sql": s, "rowcount": rowcount, "returning_df": df_ret}


# ----------------------------
# Schema fetch (tables + columns map)
# ----------------------------
@st.cache_data(ttl=600)
def get_tables_sorted(db_key: str, _engine) -> list[str]:
    insp = inspect(_engine)
    return sorted(insp.get_table_names(), key=lambda x: x.lower())

@st.cache_data(ttl=600)
def get_table_columns_map(db_key: str, _engine, tables: tuple[str, ...]) -> dict[str, list[str]]:
    insp = inspect(_engine)
    out = {}
    for t in tables:
        cols = [c["name"] for c in insp.get_columns(t)]
        out[t] = cols
    return out


# ----------------------------
# Gemini LLM — cached resource
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
        max_output_tokens=200,
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

Rules:
- Only SELECT or WITH.
- Always include LIMIT <= {max_limit}.
- If user asks "latest/newest/recent", order by id DESC (prefer id).
- If filtering on text columns (name, email, title, etc.) and value looks partial, use ILIKE with %value%.
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
# Sidebar: compact layout
# ----------------------------
with st.sidebar:
    st.caption("🗄️ Database")
    db_choice = st.selectbox(
        "DB",
        options=[
            ("Volume DB", "postgresql_volume"),
            ("Quorbit DB", "postgresql_quorbit"),
        ],
        format_func=lambda x: x[0],
        index=0,
        key="db_choice",
        label_visibility="collapsed",
    )

    db_label, db_conn_name = db_choice

    if st.session_state.active_db != db_conn_name:
        st.session_state.active_db = db_conn_name
        st.session_state.messages = []
        st.session_state.preview_table = None
        st.session_state.preview_page = 1
        st.session_state.write_logs = []
        st.session_state.open_custom_sql = False
        st.session_state.pop("selected_table", None)

        st.session_state.pop("page_size", None)
        st.session_state.pop("sort_col", None)
        st.session_state.pop("sort_dir", None)
        st.session_state.pop("page", None)

        st.toast(f"Switched to {db_label}", icon="🔄")

    cA, cB = st.columns([1, 1])
    with cA:
        edit_enabled = st.toggle("Edit", value=False, key="edit_enabled")
    with cB:
        if st.button("🧪 SQL", width='stretch'):
            st.session_state.open_custom_sql = True

    write_ok = True
    expected = st.secrets.get("WRITE_PASSCODE") or os.getenv("WRITE_PASSCODE")
    if edit_enabled and expected:
        passcode = st.text_input("Passcode", type="password", label_visibility="collapsed")
        write_ok = passcode == expected


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
table_cols_map = get_table_columns_map(st.session_state.active_db, conn.engine, tuple(table_names))

with st.sidebar:
    st.caption("📊 Tables")
    selected_table = st.selectbox(
        "Table",
        options=table_names,
        index=None,
        placeholder="Select table…",
        key="selected_table",
        label_visibility="collapsed",
    )

    if st.button("🧼 Clear screen", width='stretch'):
        st.session_state.messages = []
        st.session_state.preview_table = None
        st.session_state.preview_page = 1
        st.session_state.write_logs = []
        st.session_state.open_custom_sql = False
        st.session_state.pop("selected_table", None)
        st.toast("🧼 Cleared!", icon="🧼")
        st.rerun()


# ----------------------------
# Center screen: Custom SQL Runner panel (NO modal/dialog)
# ----------------------------
if st.session_state.open_custom_sql:
    st.markdown("---")
    st.markdown("### 🧪 Custom SQL Runner (SELECT/INSERT/UPDATE/DELETE only)")
    st.caption("DDL/admin ops blocked. UPDATE/DELETE must include WHERE.")

    sql_in = st.text_area(
        "SQL",
        height=220,
        placeholder="SELECT * FROM driver WHERE name ILIKE '%san%' LIMIT 50;\n\nUPDATE driver SET name='X' WHERE id=108 RETURNING *;"
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        run_btn = st.button("▶️ Run", width='stretch')
    with c2:
        if st.button("✖ Close", width='stretch'):
            st.session_state.open_custom_sql = False
            st.rerun()

    if run_btn:
        try:
            ok, kind, msg = validate_custom_sql(sql_in)
            if not ok:
                raise ValueError(msg)

            normalized = normalize_sql(sql_in)
            is_write = kind in ("insert", "update", "delete") or (
                kind == "with" and re.search(r"(?is)\b(insert|update|delete)\b", normalized)
            )

            if is_write and not (edit_enabled and write_ok):
                raise ValueError("Write SQL detected. Sidebar me Edit ON + passcode required (if configured).")

            if is_write:
                confirm = st.checkbox("✅ I confirm executing this INSERT/UPDATE/DELETE", value=False, key="confirm_custom_dml_inline")
                if not confirm:
                    st.info("Not executed. Tick confirmation to run.")
                    st.stop()

            out = run_custom_sql(conn, sql_in)

            st.caption("🧾 Executed SQL")
            st.code(out["sql"], language="sql")

            if out["kind"] == "select":
                st.dataframe(out["df"], width='stretch', height=360)
            else:
                st.success(f"✅ Done. Rowcount: {out.get('rowcount')}")
                if out.get("returning_df") is not None and not out["returning_df"].empty:
                    st.write("RETURNING rows:")
                    st.dataframe(out["returning_df"], width='stretch', height=300)

            st.session_state.write_logs.append({"sql": out["sql"], "params": {}})

        except Exception as e:
            st.error(f"Bhai error aa gaya 😅: {e}")


# ----------------------------
# Main: Table Preview
# ----------------------------
st.markdown("**📋 Table Preview**")

if selected_table is None:
    st.info("👈 Select a table from the left to preview data.")
else:
    safe_table_sql = quote_ident(selected_table)

    cols_df = conn.query(f"SELECT * FROM {safe_table_sql} LIMIT 0;", ttl=600)
    table_cols = list(cols_df.columns)

    pk_cols = get_primary_key_cols(conn.engine, selected_table)

    if st.session_state.preview_table != selected_table:
        st.session_state.preview_table = selected_table
        st.session_state.preview_page = 1
        st.session_state.pop("orig_page_df", None)

    with st.expander("⚙️ Preview settings", expanded=False):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            page_size = st.number_input("Page size", 10, 500, 100, step=10, key="page_size")
        with c2:
            sort_col = st.selectbox("Sort column", ["(none)"] + table_cols, index=0, key="sort_col")
        with c3:
            sort_dir = st.radio("Direction", ["ASC", "DESC"], horizontal=True, index=1, key="sort_dir")
        with c4:
            page = st.number_input("Page", 1, value=int(st.session_state.preview_page), step=1, key="page")

        st.session_state.preview_page = int(page)

        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if st.button("⬅️ Prev", width='stretch', disabled=(page <= 1), key="prev_btn"):
                st.session_state.preview_page = max(1, page - 1)
                st.rerun()
        with nav2:
            if st.button("Next ➡️", width='stretch', key="next_btn"):
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
    FROM {safe_table_sql}
    {order_clause}
    LIMIT {int(page_size)} OFFSET {int(offset)};
    """.strip()

    df_preview = conn.query(preview_sql, ttl=0)
    st.caption(f"**{selected_table}** — Rows {offset + 1} to {offset + len(df_preview)}")

    if not (edit_enabled and write_ok):
        st.dataframe(df_preview, width='stretch', height=420)
    else:
        if not pk_cols:
            st.warning("PRIMARY KEY nahi mila, isliye safe reason se edit disabled.")
            st.dataframe(df_preview, width='stretch', height=420)
        else:
            st.session_state.orig_page_df = df_preview.copy()

            df_edit = df_preview.copy()
            if "__delete__" not in df_edit.columns:
                df_edit.insert(0, "__delete__", False)

            edited = st.data_editor(
                df_edit,
                width='stretch',
                height=420,
                num_rows="fixed",
                disabled=pk_cols,
                key=f"editor_{selected_table}_{st.session_state.preview_page}",
            )
            edited = edited.applymap(to_py)

            if st.session_state.write_logs:
                with st.expander("🧾 Executed query history (latest)", expanded=False):
                    for i, item in enumerate(st.session_state.write_logs[-20:], start=1):
                        st.caption(f"#{i}")
                        st.code(item["sql"], language="sql")

            left, right = st.columns([1, 1])
            with left:
                apply_btn = st.button("✅ Apply changes", width='stretch')
            with right:
                MAX_WRITE_ROWS = 50
                st.caption(f"Safety cap: max {MAX_WRITE_ROWS} row changes per apply.")

            if apply_btn:
                try:
                    updates, deletes = diff_rows(st.session_state.orig_page_df, edited, pk_cols, delete_col="__delete__")

                    if len(updates) + len(deletes) == 0:
                        st.info("No changes detected.")
                        st.stop()

                    if len(updates) + len(deletes) > MAX_WRITE_ROWS:
                        raise ValueError(f"Too many changes ({len(updates)+len(deletes)}). Max allowed = {MAX_WRITE_ROWS}.")

                    updated_rows = []
                    deleted_rows = []
                    logs = []

                    with conn.engine.begin() as cxn:
                        del_sql = build_delete_sql(safe_table_sql, pk_cols)
                        for d in deletes:
                            params = {f"pk_{c}": to_py(d[c]) for c in pk_cols}
                            logs.append({"sql": del_sql, "params": params})
                            rows = cxn.execute(text(del_sql), params).fetchall()
                            deleted_rows.extend([dict(r._mapping) for r in rows])

                        for u in updates:
                            set_cols = u["__set_cols__"]
                            upd_sql = build_update_sql(safe_table_sql, set_cols, pk_cols)
                            params = {k: to_py(v) for k, v in u.items() if k.startswith("pk_") or k.startswith("set_")}
                            logs.append({"sql": upd_sql, "params": params})
                            rows = cxn.execute(text(upd_sql), params).fetchall()
                            updated_rows.extend([dict(r._mapping) for r in rows])

                    st.session_state.write_logs = (st.session_state.write_logs + logs)[-100:]

                    st.success(f"✅ Applied: {len(updates)} update(s), {len(deletes)} delete(s)")

                    with st.expander("🧾 Executed queries (this apply)", expanded=True):
                        for i, item in enumerate(logs, start=1):
                            st.caption(f"#{i}")
                            st.code(item["sql"], language="sql")
                            st.json(item["params"])

                    if updated_rows:
                        st.write("Updated rows (RETURNING):")
                        st.dataframe(pd.DataFrame(updated_rows), width='stretch')

                    if deleted_rows:
                        st.write("Deleted rows (RETURNING):")
                        st.dataframe(pd.DataFrame(deleted_rows), width='stretch')

                    st.rerun()

                except Exception as e:
                    st.error(f"Write failed 😅: {e}")


# ----------------------------
# Chat history
# ----------------------------
st.markdown("---")
st.markdown("**💬 Chat (Read-only)**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sql"):
            st.caption("🧾 SQL:")
            st.code(message["sql"], language="sql")


# ----------------------------
# Chat input + execution (READ ONLY) + plural fix + fk join fix + latest id fix
# ----------------------------
if prompt := st.chat_input("Ask DB (read-only)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            sql_raw = generate_sql.invoke({"question": prompt, "max_limit": str(MAX_LIMIT)})
            sql_query = extract_sql(sql_raw)

            # 1) fix plural table names
            sql_query = normalize_table_names(sql_query, table_names)

            # 2) fix common FK join mistakes like driver.driver_id -> driver.id
            sql_query = fix_bad_fk_join(sql_query, table_names, table_cols_map)

            # 3) prefer id desc and fix ORDER BY id ambiguity on joins
            sql_query = prefer_id_order_for_latest(sql_query, table_names, table_cols_map)

            if not is_safe_select(sql_query):
                raise ValueError(f"Unsafe SQL:\n{sql_query}")

            sql_query = enforce_limit(sql_query, MAX_LIMIT)

            st.caption("🧾 SQL")
            st.code(sql_query, language="sql")

            df = conn.query(sql_query, ttl=0)
            st.dataframe(df, width='stretch')

            raw_rows = df.head(10).to_dict(orient="records")
            response = answer_chain.invoke({"question": prompt, "response": raw_rows})
            st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response, "sql": sql_query})

        except Exception as e:
            response = f"Bhai error aa gaya 😅: {e}"
            st.error(response)
            st.session_state.messages.append({"role": "assistant", "content": response})


st.markdown("---")
st.caption("Made with ❤️ | Gemini + PostgreSQL + Streamlit")
