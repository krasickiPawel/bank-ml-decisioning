# streamlit_app.py
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

import requests
import streamlit as st
import pandas as pd

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk Demo", layout="wide")
st.title("Credit Risk Decisioning — demo")


SAFE: Dict[str, Any] = {
    "checking_status": ">=200",
    "duration": 12,
    "credit_history": "all paid",
    "purpose": "radio/tv",
    "credit_amount": 1500,
    "savings_status": ">=1000",
    "employment": ">=7",
    "installment_commitment": 2,
    "personal_status": "male single",
    "other_parties": "none",
    "residence_since": 3,
    "property_magnitude": "real estate",
    "age": 40,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "yes",
    "foreign_worker": "yes",
}

RISKY: Dict[str, Any] = {
    "checking_status": "<0",
    "duration": 48,
    "credit_history": "critical/other existing credit",
    "purpose": "car (used)",
    "credit_amount": 9000,
    "savings_status": "no known savings",
    "employment": "unemployed",
    "installment_commitment": 4,
    "personal_status": "female div/dep/mar",
    "other_parties": "guarantor",
    "residence_since": 1,
    "property_magnitude": "no known property",
    "age": 23,
    "other_payment_plans": "bank",
    "housing": "rent",
    "existing_credits": 2,
    "job": "unskilled resident",
    "num_dependents": 2,
    "own_telephone": "none",
    "foreign_worker": "yes",
}

BORDER: Dict[str, Any] = {
    "checking_status": "0<=X<200",
    "duration": 24,
    "credit_history": "existing paid",
    "purpose": "furniture/equipment",
    "credit_amount": 3500,
    "savings_status": "100<=X<500",
    "employment": "1<=X<4",
    "installment_commitment": 3,
    "personal_status": "male mar/wid",
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "car",
    "age": 30,
    "other_payment_plans": "none",
    "housing": "free",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "none",
    "foreign_worker": "yes",
}


# -------------------------
# HTTP helpers (never crash Streamlit)
# -------------------------
def render_citations(citations: list[dict]):
    if not citations:
        st.info("No citations returned.")
        return

    for i, c in enumerate(citations, start=1):
        score = c.get("score")
        source = c.get("source", "unknown")
        chunk_id = c.get("chunk_id", "unknown")
        text = c.get("text", "")

        with st.expander(f"[{i}] {source} • {chunk_id} • score={score:.3f}" if isinstance(score, (int, float)) else f"[{i}] {source} • {chunk_id}"):
            st.code(text)


def _safe_json(resp: requests.Response) -> Dict[str, Any]:
    ct = (resp.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        try:
            return resp.json()
        except Exception:
            return {"status": "error", "http_status": resp.status_code, "body": resp.text[:2000]}
    return {"status": "error", "http_status": resp.status_code, "body": resp.text[:2000]}


def post_json(url: str, payload=None, timeout: int = 60):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return _safe_json(r)
    except requests.exceptions.Timeout:
        return {"status": "error", "message": f"Request timeout after {timeout}s"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def http_get(url: str, timeout: int = 30, params: Optional[dict] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        data = _safe_json(resp)
        if resp.status_code != 200:
            return data, f"HTTP {resp.status_code}", resp.status_code
        return data, None, resp.status_code
    except requests.exceptions.Timeout:
        return None, f"Request timeout after {timeout}s", 0
    except Exception as e:
        return None, str(e), 0


def http_post(url: str, timeout: int = 60, payload: Optional[dict] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str], int]:
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        data = _safe_json(resp)
        if resp.status_code != 200:
            return data, f"HTTP {resp.status_code}", resp.status_code
        return data, None, resp.status_code
    except requests.exceptions.Timeout:
        return None, f"Request timeout after {timeout}s", 0
    except Exception as e:
        return None, str(e), 0


def wait_for_api_idle(api_url: str, max_wait_s: int = 20, poll_s: float = 0.8) -> Dict[str, Any]:
    """
    Poll /health until reload_status becomes idle (or error / timeout).
    """
    deadline = time.time() + max_wait_s
    last: Dict[str, Any] = {"status": "error", "message": "No health response yet."}

    while time.time() < deadline:
        data, err, _ = http_get(f"{api_url}/health", timeout=3)
        if data:
            last = data
            rs = str(data.get("reload_status", "idle"))
            if rs == "idle":
                return data
            if rs == "error":
                return data
        time.sleep(poll_s)

    last["status"] = "error"
    last["message"] = f"Timeout waiting for reload_status=idle (waited {max_wait_s}s)."
    return last


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("API")
api_url = st.sidebar.text_input("API_URL", DEFAULT_API_URL)

health_box = st.sidebar.empty()

col_a, col_b = st.sidebar.columns(2)

if col_a.button("Health"):
    data, err, _ = http_get(f"{api_url}/health", timeout=3)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.json(data)

if col_b.button("Reload (demo)"):
    data, err, _ = http_post(f"{api_url}/reload", timeout=3)
    if err:
        st.sidebar.error(err)
    else:
        st.sidebar.info("Reload requested. Waiting for API to become idle...")
        with st.sidebar:
            with st.spinner("Reloading model..."):
                final = wait_for_api_idle(api_url, max_wait_s=25, poll_s=0.8)
        st.sidebar.json(final)

# Always show current /health snapshot (non-blocking)
health_data, health_err, _ = http_get(f"{api_url}/health", timeout=2)
if health_err:
    health_box.error(health_err)
else:
    rs = str(health_data.get("reload_status", "idle"))
    clf = str(health_data.get("clf_name", "unknown"))
    run_id = str(health_data.get("run_id") or "")
    thr = health_data.get("threshold", None)
    if rs in {"loading", "reloading"}:
        health_box.warning(f"API status: {rs} | clf={clf}")
    elif rs == "error":
        health_box.error(f"API status: error | clf={clf}")
    else:
        if thr is None:
            health_box.success(f"API status: idle | clf={clf} | run={run_id[:8]}")
        else:
            health_box.success(f"API status: idle | clf={clf} | run={run_id[:8]} | thr={thr:.2f}")


tabs = st.tabs(["Demo", "Monitoring", "RAG"])


# -------------------------
# Demo tab
# -------------------------
with tabs[0]:
    st.subheader("Scenario")
    scenario = st.selectbox("Preset", ["SAFE", "RISKY", "BORDER", "MINIMAL"])
    if scenario == "SAFE":
        base = SAFE
    elif scenario == "RISKY":
        base = RISKY
    elif scenario == "BORDER":
        base = BORDER
    else:
        base = {"duration": 12, "credit_amount": 1000, "age": 35}

    schema_data, schema_err, schema_status = http_get(f"{api_url}/schema", timeout=10)

    if schema_status == 503:
        st.warning("API is loading/reloading. Try again in a moment.")
        if schema_data:
            st.json(schema_data)
        st.stop()

    if schema_err:
        st.error(f"/schema error: {schema_err}")
        if schema_data:
            st.json(schema_data)
        st.stop()

    expected_cols = schema_data.get("expected_columns", [])
    st.caption(f"Expected columns: {len(expected_cols)}")

    # Prefer server-provided example if present
    server_example = schema_data.get("example_payload", {}).get("features", {})
    if isinstance(server_example, dict) and server_example:
        default_features = dict(server_example)
        default_features.update(base)  # scenario overrides a few fields for demo
    else:
        default_features = dict(base)

    st.subheader("Client features (JSON)")
    features_text = st.text_area("features", value=json.dumps(default_features, indent=2), height=280)

    top_k = st.slider("Top K reason codes", 0, 8, 3)

    if st.button("Predict"):
        try:
            features_obj = json.loads(features_text)
            if not isinstance(features_obj, dict):
                raise ValueError("features JSON must be an object/dict")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        payload = {"features": features_obj, "top_k_reasons": int(top_k)}
        out, err, status = http_post(f"{api_url}/predict", payload=payload, timeout=60)

        if status == 503:
            st.warning("API is loading/reloading. Try again in a moment.")
            if out:
                st.json(out)
            st.stop()

        if err:
            st.error(f"/predict error: {err}")
            if out:
                st.json(out)
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score (P bad)", f"{out.get('score', 0.0):.3f}")
        c2.metric("Decision", str(out.get("decision", "n/a")))
        c3.metric("Threshold", f"{out.get('threshold', 0.0):.2f}")
        rid = str(out.get("run_id", ""))
        c4.metric("Run ID", rid[:8] if rid else "n/a")

        reasons = out.get("reasons", [])
        if reasons:
            st.subheader("Reason codes")
            st.dataframe(reasons, use_container_width=True)
        else:
            st.info("No reason codes returned for this classifier (or top_k=0).")

        st.subheader("Raw response")
        st.json(out)


# -------------------------
# Monitoring tab
# -------------------------
with tabs[1]:
    st.subheader("Drift (PSI/KS) based on request logs")
    limit = st.slider("Requests window", 50, 500, 200, step=50)

    if st.button("Refresh drift"):
        data, err, status = http_get(f"{api_url}/metrics/drift", params={"limit": limit}, timeout=30)

        if status == 503:
            st.warning("API is loading/reloading. Try again in a moment.")
            if data:
                st.json(data)
            st.stop()

        if err:
            st.error(f"/metrics/drift error: {err}")
            if data:
                st.json(data)
            st.stop()

        if not data:
            st.error("No response.")
            st.stop()

        if data.get("status") == "error":
            st.warning(data.get("message", "Drift returned status=error."))
            st.json(data)
            st.stop()

        ref_src = data.get("reference_source", "unknown")
        st.caption(f"Reference source: {ref_src}")

        results = data.get("results", [])
        if not results:
            st.info("No drift results (need more requests, or reference not available).")
            st.json(data)
            st.stop()

        # Show a quick ranked table
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)

        st.subheader("Raw response")
        st.json(data)


with tabs[2]:
    st.subheader("RAG: Ask the docs")
    question = st.text_input("Question", value="How does threshold work in this demo?")
    top_k = st.slider("Top K context chunks", 1, 10, 4)

    if st.button("ASK"):
        payload = {"question": question, "top_k": int(top_k)}
        res = post_json(f"{api_url}/rag/ask", payload=payload, timeout=60)

        if res.get("status") != "ok":
            st.error("RAG error")
            st.json(res)
        else:
            st.caption(f"Indexed run_id: {res.get('indexed_run_id')}")
            st.markdown("### Answer (extractive)")
            st.code(res.get("answer", ""))

            st.markdown("### Citations")
            render_citations(res.get("citations", []))
    # st.subheader("RAG: Ask the docs")
    # q = st.text_input("Question", "How does threshold work in this demo?")
    # top_k = st.slider("Top K context chunks", 1, 8, 4)
    #
    # if st.button("Ask"):
    #     out = post_json(f"{api_url}/rag/answer", {"question": q, "top_k": int(top_k)}, timeout=60)
    #     st.write(out.get("answer", ""))
    #     st.divider()
    #     st.caption("Citations")
    #     st.json(out.get("citations", []))