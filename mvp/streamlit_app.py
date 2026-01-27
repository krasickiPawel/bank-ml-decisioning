"""
MVP: minimalny UI do Credit Risk API. Uruchom: streamlit run streamlit_app.py
"""
import os
import streamlit as st
import requests

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Credit Risk MVP", layout="centered")
st.title("Credit Risk MVP — demo")

api_url = st.text_input("API_URL", value=DEFAULT_API_URL, key="api_url")

# Health
if st.button("Health"):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        d = r.json()
        st.success(f"Status: {d.get('status', '?')} | threshold={d.get('threshold')} | clf={d.get('clf_name')}")
    except requests.exceptions.Timeout:
        st.error("Timeout (API nie odpowiada w 5 s)")
    except Exception as e:
        st.error(str(e))

# Schema
if st.button("Schema"):
    try:
        r = requests.get(f"{api_url.rstrip('/')}/schema", timeout=5)
        d = r.json()
        st.json({"expected_columns": d.get("expected_columns"), "example_payload": d.get("example_payload")})
    except requests.exceptions.Timeout:
        st.error("Timeout")
    except Exception as e:
        st.error(str(e))

# Predict — przykładowe features (skrócona lista)
st.subheader("Predict")
example = {
    "checking_status": "<0",
    "duration": 12,
    "credit_amount": 1000,
    "age": 35,
    "installment_commitment": 2,
    "residence_since": 3,
    "existing_credits": 1,
    "num_dependents": 0,
    "credit_history": "all paid",
    "purpose": "radio/tv",
    "savings_status": ">=1000",
    "employment": ">=7",
    "personal_status": "male single",
    "other_parties": "none",
    "property_magnitude": "real estate",
    "other_payment_plans": "none",
    "housing": "own",
    "job": "skilled",
    "own_telephone": "yes",
    "foreign_worker": "yes",
}
if st.button("Predict (przykład)"):
    try:
        r = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json={"features": example},
            timeout=5,
        )
        d = r.json()
        st.success(f"Score: {d.get('score', 0):.4f} | Decision: {d.get('decision')} | Threshold: {d.get('threshold')}")
    except requests.exceptions.Timeout:
        st.error("Timeout")
    except Exception as e:
        st.error(str(e))
