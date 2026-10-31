"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline: intent classification -> verified template OR generated SQL
          -> 5-check validation gate -> (one retry) -> execute (read-only)
          -> narrative response + SQL + raw data + confidence + audit log.
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import sqlparse  # noqa: F401  (kept from the notebook; used for statement parsing)
import streamlit as st
from langchain_openai import ChatOpenAI

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk Query Engine", page_icon="🏦", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "credit_risk_portfolio.db"          # must sit next to app.py
TEST_CSV_PATH = APP_DIR / "test_queries.csv"            # optional
AUDIT_LOG_PATH = APP_DIR / "audit_log.jsonl"            # ephemeral on Streamlit Cloud


# --------------------------------------------------------------------------
# Credentials / LLM setup
# --------------------------------------------------------------------------
def _get_secret(name, default=None):
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


@st.cache_resource(show_spinner=False)
def get_llms():
    api_key = _get_secret("OPENAI_API_KEY")
    api_base = _get_secret("OPENAI_API_BASE")
    model = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        return None, None
    kwargs = dict(model=model, api_key=api_key)
    if api_base:
        kwargs["base_url"] = api_base
    # Generation / routing / narrative LLM: low temperature for repeatability
    llm = ChatOpenAI(temperature=0, **kwargs)
    # Separate evaluator LLM used only for the relevance check
    evaluator_llm = ChatOpenAI(temperature=0, **kwargs)
    return llm, evaluator_llm


# --------------------------------------------------------------------------
# Database (read-only)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection():
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Add credit_risk_portfolio.db next to app.py."
        )
    # Read-only URI connection prevents any write operation.
    # check_same_thread=False because Streamlit runs scripts in worker threads.
    return sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True, check_same_thread=False)


# --------------------------------------------------------------------------
# Database schema supplied to the LLM
# --------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# --------------------------------------------------------------------------
# Verified Query Template Library
# NOTE: VQ5-VQ10 in the notebook contained markdown fences and referenced
# columns that do not exist (loan_master.sector, current_internal_rating).
# They are corrected here so the templates run as written.
# --------------------------------------------------------------------------
sql_1 = """
SELECT
    s.sector_name,
    SUM(l.total_outstanding) / 1000000.0 AS total_outstanding_millions,
    SUM(
        CASE
            WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
            THEN l.total_outstanding
            ELSE 0
        END
    ) / 1000000.0 AS npa_exposure_millions
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
GROUP BY s.sector_code, s.sector_name
ORDER BY total_outstanding_millions DESC;
"""

sql_2 = """
SELECT
    loan_category,
    COUNT(*) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS outstanding_exposure_millions
FROM loan_master
GROUP BY loan_category
ORDER BY outstanding_exposure_millions DESC;
"""

sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    SUM(ead_amount) / 1000000.0 AS total_ead_millions,
    SUM(ecl_amount) / 1000000.0 AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;
"""

sql_4 = """
SELECT
    s.sector_name,
    AVG(p.provision_coverage_ratio) AS average_provision_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_code, s.sector_name
ORDER BY average_provision_coverage_ratio DESC;
"""

sql_5 = """
SELECT
    l.borrower_name,
    s.sector_name AS sector,
    l.total_outstanding / 1000000.0 AS outstanding_exposure_millions,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10;
"""

sql_6 = """
SELECT
    group_name,
    COUNT(*) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_exposure_millions
FROM loan_master
WHERE group_name IS NOT NULL
  AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY total_exposure_millions DESC
LIMIT 5;
"""

sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name AS sector,
    l.total_outstanding / 1000000.0 AS outstanding_amount_millions,
    l.days_past_due,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

sql_8 = """
SELECT
    dpd_bucket,
    COUNT(*) AS loan_count,
    SUM(total_outstanding) / 1000000.0 AS total_outstanding_millions
FROM (
    SELECT
        total_outstanding,
        CASE
            WHEN days_past_due = 0 THEN '0 (current)'
            WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
            WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
            WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
            WHEN days_past_due > 90 THEN '90+'
        END AS dpd_bucket,
        CASE
            WHEN days_past_due = 0 THEN 1
            WHEN days_past_due BETWEEN 1 AND 30 THEN 2
            WHEN days_past_due BETWEEN 31 AND 60 THEN 3
            WHEN days_past_due BETWEEN 61 AND 90 THEN 4
            WHEN days_past_due > 90 THEN 5
        END AS bucket_order
    FROM loan_master
    WHERE days_past_due >= 0
) AS bucketed_loans
GROUP BY dpd_bucket, bucket_order
ORDER BY bucket_order;
"""

sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating AS current_internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

sql_10 = """
SELECT
    reporting_date,
    SUM(ecl_amount) / 1000000.0 AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;
"""

verified_query_library = {
    "VQ1": {"description": "Sector-wise total outstanding and NPA amount breakdown across all sectors", "sql": sql_1},
    "VQ2": {"description": "Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)", "sql": sql_2},
    "VQ3": {"description": "IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter", "sql": sql_3},
    "VQ4": {"description": "Average provision coverage ratio by sector for the latest reporting quarter", "sql": sql_4},
    "VQ5": {"description": "Top 10 largest loan exposures by outstanding amount at the borrower level", "sql": sql_5},
    "VQ6": {"description": "Top 5 largest exposures aggregated at the business group level", "sql": sql_6},
    "VQ7": {"description": "All overdue loan accounts with their days past due and asset classification", "sql": sql_7},
    "VQ8": {"description": "Distribution of loans across days-past-due buckets showing aging profile of the portfolio", "sql": sql_8},
    "VQ9": {"description": "Borrowers whose internal rating was downgraded in the latest rating cycle", "sql": sql_9},
    "VQ10": {"description": "Expected credit loss trend across all reporting quarters showing provisioning movement over time", "sql": sql_10},
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def extract_json(text):
    """Pull the first JSON object out of an LLM response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def clean_sql(sql):
    """Strip markdown fences and surrounding whitespace."""
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql.strip(), flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql, flags=re.MULTILINE).strip()
    return sql


# --------------------------------------------------------------------------
# Tool 1: Intent classification
# --------------------------------------------------------------------------
def classify_intent(llm, user_question, query_library):
    library_descriptions = "\n".join(
        f"{qid}: {entry['description']}" for qid, entry in query_library.items()
    )

    prompt = f"""You are the routing component of a commercial-lending credit-risk query engine.

Decide whether the user's question can be answered by ONE of the pre-approved verified
query templates listed below, or whether new SQL must be generated.

Rules:
- Choose route "verified" ONLY if a template's underlying data and logic would contain
  everything needed to answer the question. A template may return more rows than the user
  asked about (for example all sectors when the user asks about one sector); that is
  acceptable because a later step highlights the relevant rows.
- Choose route "generated" if the question needs a metric, grouping, filter, time period,
  or table combination that none of the templates provide (for example a different
  breakdown, restructured-loan analysis, interest-rate analysis, or a trend of a metric
  the templates do not trend).
- Pick at most one template. Never invent a template ID.

User question:
{user_question}

Verified query templates:
{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""
    response = llm.invoke(prompt).content.strip()
    parsed = extract_json(response)
    if parsed and parsed.get("route") in ("verified", "generated"):
        if parsed["route"] == "verified" and parsed.get("query_id") not in query_library:
            return {"route": "generated", "query_id": None,
                    "match_reason": "Classifier returned an unknown template ID; falling back to generation"}
        return parsed
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# --------------------------------------------------------------------------
# Tool 2: Query generation
# --------------------------------------------------------------------------
def generate_query(llm, user_question, schema_context):
    prompt = f"""You are a senior credit-risk data analyst writing SQL for a SQLite database
that holds a commercial lending portfolio.

Write ONE read-only SQLite query that answers the user's question.

Hard rules:
- Output a single SELECT statement (a WITH ... SELECT is allowed). Never write INSERT,
  UPDATE, DELETE, DROP, ALTER, REPLACE, ATTACH, PRAGMA or any other write/DDL statement.
- Use ONLY the tables and columns in the schema below. Do not invent columns.
- To get a sector name, join loan_master.sector_code to sector_master.sector_code.
- Amounts are in USD; present exposure/ECL/EAD in millions (divide by 1000000.0) and name
  the column with a "_millions" suffix.
- NPA means asset_classification IN ('Substandard', 'Doubtful', 'Loss').
- "Latest" quarter means reporting_date = '2025-09-30' in provisioning and
  rating_date = '2025-09-30' in borrower_rating. For trends use all available dates.
- Restructured loans: is_restructured = 1.
- Give every computed column a clear alias and sort results sensibly.
- Return ONLY the SQL. No explanation and no markdown fences.

User question:
{user_question}

Database schema:
{schema_context}
"""
    return clean_sql(llm.invoke(prompt).content)


# --------------------------------------------------------------------------
# Tool 3: Validation (5 checks)
# --------------------------------------------------------------------------
SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "group", "by", "order", "having", "limit", "join", "on", "as",
    "case", "when", "then", "else", "end", "sum", "count", "avg", "min", "max", "round", "desc", "asc",
    "left", "right", "inner", "outer", "distinct", "null", "is", "not", "in", "like", "with", "union",
    "all", "between", "coalesce", "cast", "real", "integer", "text", "ifnull", "abs", "length", "trim",
    "substr", "date", "strftime", "now", "over", "partition", "row_number", "rank", "cross", "using",
    "exists", "offset", "iif", "nullif", "lower", "upper",
}


def validate_query(llm_eval, user_question, candidate_sql, db_connection, query_library, query_id=None):
    result = {"passed": False, "failed_check": None, "details": "", "relevance_confidence": None}

    # Strip string literals so keywords/identifiers inside quotes aren't misjudged
    sql_no_strings = re.sub(r"'(?:[^']|'')*'", "''", candidate_sql)
    sql_upper = sql_no_strings.upper().strip()

    # ---- Check 1: read-only shape ----
    forbidden_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE",
                          "ATTACH", "DETACH", "PRAGMA", "CREATE", "VACUUM"]
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        result.update(failed_check="read_only_shape", details="Query must start with SELECT or WITH")
        return result
    for kw in forbidden_keywords:
        if re.search(r"\b" + kw + r"\b", sql_upper):
            result.update(failed_check="read_only_shape", details=f"Forbidden keyword detected: {kw}")
            return result
    if ";" in sql_no_strings.strip().rstrip(";"):
        result.update(failed_check="read_only_shape", details="Multiple statements are not allowed")
        return result

    # ---- Check 2: schema conformance ----
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f'PRAGMA table_info("{t}")').fetchall():
            real_columns.add(col_info[1].lower())
    real_tables_lower = {t.lower() for t in real_tables}

    # Aliases (after AS) and bare table aliases are legitimate identifiers
    aliases = set(re.findall(r"\bas\s+([a-z_][a-z0-9_]*)", sql_no_strings.lower()))
    aliases |= set(re.findall(r"\b(?:from|join)\s+[a-z_][a-z0-9_]*\s+(?!on\b|where\b|group\b|order\b|join\b|left\b|inner\b|limit\b|using\b)([a-z_][a-z0-9_]*)",
                              sql_no_strings.lower()))
    identifiers = re.findall(r"\b[a-z_][a-z0-9_]*\b", sql_no_strings.lower())
    unknown = sorted({
        tok for tok in identifiers
        if tok not in SQL_KEYWORDS and tok not in real_columns and tok not in real_tables_lower
        and tok not in aliases
    })
    if unknown:
        result.update(failed_check="schema_conformance",
                      details=f"Unknown tables/columns referenced: {', '.join(unknown)}")
        return result

    # ---- Check 3: parse and plan ----
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result.update(failed_check="parse_plan_dry_run", details=f"SQL failed to parse or plan: {e}")
        return result

    # ---- Check 4: LLM relevance ----
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad (e.g., it may "
        "return all sectors/categories/stages rather than filtering to just what the user asked). "
        "A separate response-generation step will filter and highlight the relevant rows afterward. "
        "Do NOT fail this query for lacking a WHERE clause that narrows to the user's specific "
        "sector/category/stage - judge only whether the underlying metric, tables, and aggregation "
        "logic match the question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be appropriately "
        "scoped/filtered to answer it directly."
    )
    relevance_prompt = f"""You are a strict reviewer of SQL written for a commercial-lending credit-risk database.

Judge whether the SQL correctly answers the user's question. Check that:
- the right tables and joins are used,
- the metric and aggregation match the question (e.g. total_outstanding for exposure,
  ecl_amount for expected credit loss),
- NPA logic uses asset_classification IN ('Substandard', 'Doubtful', 'Loss') where relevant,
- date logic uses the correct date column and the latest date '2025-09-30' when the
  question asks about the latest/current quarter.

{track_context}

User question:
{user_question}

SQL:
{candidate_sql}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    relevance_json = extract_json(evaluator_text := llm_eval.invoke(relevance_prompt).content.strip())
    if relevance_json is None:
        result.update(failed_check="llm_relevance", details="Relevance check returned an unparseable response")
        return result
    try:
        confidence = float(relevance_json.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    result["relevance_confidence"] = confidence
    if str(relevance_json.get("verdict", "")).lower() == "no" or confidence < 0.6:
        result.update(failed_check="llm_relevance",
                      details=f"Relevance check failed: {relevance_json.get('reason', 'unknown')}")
        return result

    # ---- Check 5: template integrity (verified track) ----
    if is_verified_track:
        expected_sql = query_library[query_id]["sql"].strip().rstrip(";")
        cand = candidate_sql.strip().rstrip(";")
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({cand}) LIMIT 0").description]
            if expected_cols != actual_cols:
                result.update(failed_check="template_integrity",
                              details=f"Expected columns {expected_cols}, got {actual_cols}")
                return result
        except sqlite3.Error as e:
            result.update(failed_check="template_integrity", details=f"Template integrity check failed: {e}")
            return result

    result["passed"] = True
    result["details"] = "All validation checks passed"
    return result


# --------------------------------------------------------------------------
# Tool 4: Retry generation
# --------------------------------------------------------------------------
def retry_generation(llm, user_question, failed_sql, error_message, schema_context):
    prompt = f"""You are a senior credit-risk data analyst. A SQLite query you wrote failed validation.
Fix it so it passes while still answering the user's original question.

Rules:
- Output a single read-only SELECT (or WITH ... SELECT) statement. No write/DDL statements.
- Use ONLY tables and columns present in the schema. Do not invent columns.
- Address the specific validation error below.
- Return ONLY the corrected SQL, with no explanation and no markdown fences.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}
"""
    return clean_sql(llm.invoke(prompt).content)


# --------------------------------------------------------------------------
# Tool 5: Execution
# --------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    result = {"dataframe": None, "reasonable": True, "warnings": []}
    df = pd.read_sql_query(validated_sql, db_connection)
    result["dataframe"] = df

    if df.empty:
        result["warnings"].append("Query returned an empty result")

    for col in df.select_dtypes(include="number").columns:
        if (df[col] < 0).any() and "deviation" not in col.lower() and "change" not in col.lower():
            result["warnings"].append(f"Column {col} contains negative values")
        if df[col].isnull().any():
            null_count = int(df[col].isnull().sum())
            if null_count > len(df) * 0.5:
                result["warnings"].append(f"Column {col} has {null_count} null values")

    if len(result["warnings"]) > 2:
        result["reasonable"] = False
    return result


# --------------------------------------------------------------------------
# Tool 6: Response generation
# --------------------------------------------------------------------------
def generate_response(llm, user_question, dataframe, route, query_id=None):
    prompt = f"""You are a credit-risk analyst writing a short answer for a business user at a commercial bank.

Answer the user's question using ONLY the query result below.

Guidelines:
- Lead with the direct answer, then add at most 2-3 supporting observations.
- Focus on what was asked. If the result covers more rows than the question needs
  (for example all sectors when one was asked about), highlight only the relevant rows.
- Quote exact figures from the data. Do not invent, estimate, or round away precision.
  Amounts labelled "_millions" are in USD millions.
- If the result is empty, say so plainly.
- Do not mention SQL, tables, or the pipeline. Keep it under 150 words.

Route: {route}{f' ({query_id})' if query_id else ''}

User question:
{user_question}

Query result:
{dataframe.to_string(index=False)}
"""
    return llm.invoke(prompt).content.strip()


# --------------------------------------------------------------------------
# Audit trail
# --------------------------------------------------------------------------
def write_audit(entry):
    record = {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry}
    try:
        with AUDIT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        pass  # filesystem may be read-only; session log below still works
    st.session_state.setdefault("audit", []).append(record)


# --------------------------------------------------------------------------
# Pipeline orchestration
# --------------------------------------------------------------------------
def run_pipeline(llm, llm_eval, user_question, db_connection, query_library, schema_context):
    log = {
        "user_question": user_question, "route": None, "query_id": None, "match_reason": None,
        "candidate_sql": None, "gate_result": None, "retry_used": False, "escalated": False,
        "executed_sql": None, "row_count": None, "confidence": None, "narrative": None,
    }
    steps = []

    # 1. Intent classification
    classification = classify_intent(llm, user_question, query_library)
    log["route"] = classification["route"]
    log["query_id"] = classification.get("query_id")
    log["match_reason"] = classification.get("match_reason")
    steps.append(f"Intent classification -> route **{log['route']}**, query **{log['query_id']}**. {log['match_reason']}")

    # 2. Query construction
    if log["route"] == "verified" and log["query_id"] in query_library:
        candidate_sql = query_library[log["query_id"]]["sql"].strip()
        steps.append("Query construction -> loaded verified template from library")
    else:
        candidate_sql = generate_query(llm, user_question, schema_context)
        steps.append("Query construction -> generated new SQL")
    log["candidate_sql"] = candidate_sql

    # 3. Validation gate
    gate = validate_query(llm_eval, user_question, candidate_sql, db_connection, query_library, log["query_id"])
    log["gate_result"] = gate
    steps.append(f"Validation gate -> passed={gate['passed']}, relevance confidence={gate.get('relevance_confidence')}"
                 + ("" if gate["passed"] else f" | failed check: {gate['failed_check']} ({gate['details']})"))

    # 4. One retry on the generated track
    if not gate["passed"] and log["route"] == "generated":
        steps.append(f"Retrying once: {gate['details']}")
        candidate_sql = retry_generation(llm, user_question, candidate_sql, gate["details"], schema_context)
        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True
        gate = validate_query(llm_eval, user_question, candidate_sql, db_connection, query_library, None)
        log["gate_result"] = gate
        steps.append(f"Retry validation gate -> passed={gate['passed']}, relevance confidence={gate.get('relevance_confidence')}"
                     + ("" if gate["passed"] else f" | failed check: {gate['failed_check']} ({gate['details']})"))

    # 5. Escalate
    if not gate["passed"]:
        log["escalated"] = True
        log["confidence"] = "ESCALATED"
        log["narrative"] = ("This question could not be resolved reliably and has been escalated to a "
                            f"human analyst. Reason: {gate['details']}")
        steps.append("Escalated to human analyst")
        return {**log, "dataframe": None, "steps": steps, "warnings": []}

    # 6. Execute
    log["executed_sql"] = candidate_sql
    try:
        exec_result = execute_query(candidate_sql, db_connection)
    except Exception as e:  # execution failure also escalates
        log["escalated"] = True
        log["confidence"] = "ESCALATED"
        log["narrative"] = f"Execution failed and the question was escalated to a human analyst. Error: {e}"
        steps.append(f"Execution error -> {e}")
        return {**log, "dataframe": None, "steps": steps, "warnings": []}

    df = exec_result["dataframe"]
    log["row_count"] = len(df)
    steps.append(f"Execution -> {len(df)} rows returned"
                 + (f" | warnings: {exec_result['warnings']}" if exec_result["warnings"] else ""))

    # 7. Narrative
    log["narrative"] = generate_response(llm, user_question, df, log["route"], log["query_id"])
    log["confidence"] = gate.get("relevance_confidence")
    steps.append(f"Response generation -> confidence {log['confidence']}")

    return {**log, "dataframe": df, "steps": steps, "warnings": exec_result["warnings"]}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
EXAMPLE_QUESTIONS = [
    "Which sectors have the highest NPA exposure?",
    "Show me the DPD aging profile of the portfolio.",
    "Which restructured loans are also classified as impaired?",
    "What is the average interest rate by sector?",
    "How has Stage 3 exposure and ECL trended over the last four quarters?",
    "Who are our top 5 business groups by exposure?",
    "Which borrowers were downgraded in the latest rating cycle?",
]


def render_result(res):
    if res["escalated"]:
        st.error("🚩 Escalated to a human analyst")
        st.write(res["narrative"])
    else:
        conf = res["confidence"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", res["route"])
        c2.metric("Query ID", res["query_id"] or "Generated")
        c3.metric("Confidence", f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a")
        c4.metric("Rows", res["row_count"])
        st.subheader("Answer")
        st.write(res["narrative"])
        for w in res["warnings"]:
            st.warning(w)

    with st.expander("SQL used", expanded=not res["escalated"]):
        st.code(res["executed_sql"] or res["candidate_sql"], language="sql")
    if res["dataframe"] is not None:
        with st.expander("Raw data returned", expanded=True):
            st.dataframe(res["dataframe"], use_container_width=True)
            st.download_button("Download CSV", res["dataframe"].to_csv(index=False),
                               file_name="query_result.csv", mime="text/csv")
    with st.expander("Pipeline steps and validation details"):
        for s in res["steps"]:
            st.markdown(f"- {s}")
        st.json({"validation": res["gate_result"], "retry_used": res["retry_used"]})


def main():
    st.title("🏦 Credit Risk Query Engine")
    st.caption("Northbridge Bank | Commercial lending portfolio | Read-only analytics. "
               "Every answer shows the SQL, the raw data, and a confidence score.")

    llm, llm_eval = get_llms()
    if llm is None:
        st.error("OPENAI_API_KEY is not configured. Add it to Streamlit secrets "
                 "(and OPENAI_API_BASE if you use a custom endpoint).")
        st.stop()
    try:
        conn = get_connection()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.sidebar:
        st.header("Verified query library")
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}** - {entry['description']}")
        st.divider()
        st.header("Try an example")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state["question"] = q
        st.divider()
        audit = st.session_state.get("audit", [])
        st.caption(f"Audit entries this session: {len(audit)}")
        if audit:
            st.download_button("Download audit log (JSONL)",
                               "\n".join(json.dumps(a, default=str) for a in audit),
                               file_name="audit_log.jsonl", mime="application/json")

    question = st.text_area("Ask a question about the commercial lending portfolio",
                            key="question", height=90,
                            placeholder="e.g. Which sectors have the highest NPA exposure?")

    if st.button("Run query", type="primary") and question.strip():
        with st.spinner("Running pipeline..."):
            try:
                res = run_pipeline(llm, llm_eval, question.strip(), conn, verified_query_library, database_schema)
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()
        st.session_state["last_result"] = res
        write_audit({
            "user_question": res["user_question"], "route": res["route"], "query_id": res["query_id"],
            "match_reason": res["match_reason"], "sql": res["executed_sql"] or res["candidate_sql"],
            "retry_used": res["retry_used"], "escalated": res["escalated"], "row_count": res["row_count"],
            "confidence": res["confidence"], "validation": res["gate_result"], "narrative": res["narrative"],
        })

    if "last_result" in st.session_state:
        render_result(st.session_state["last_result"])


if __name__ == "__main__":
    main()
