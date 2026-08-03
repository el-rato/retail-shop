"""Streamlit dashboard for the Smart Retail platform.

Reads aggregated statistics from the FastAPI backend (GET /dashboard/stats)
and renders customer visits, sentiment trends, chatbot logs, and product
predictions. Also exposes small interactive widgets to call the ML endpoints.

Run:
    streamlit run frontend/dashboard.py
"""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.getenv("DASHBOARD_API_BASE") or st.secrets.get("api_base", "http://localhost:8000")
API_KEY = os.getenv("API_KEY") or st.secrets.get("api_key", "change-me-in-production")
HEADERS = {"X-API-Key": API_KEY}


def _get(path: str) -> Any:
    with httpx.Client(base_url=API_BASE, headers=HEADERS, timeout=15) as client:
        resp = client.get(path)
        resp.raise_for_status()
        return resp.json()


def _post(path: str, payload: dict) -> Any:
    with httpx.Client(base_url=API_BASE, headers=HEADERS, timeout=30) as client:
        resp = client.post(path, json=payload)
        resp.raise_for_status()
        return resp.json()


def _image_to_base64(uploaded) -> str:
    data = uploaded.getvalue()
    return base64.b64encode(data).decode("utf-8")


st.set_page_config(page_title="Smart Retail Dashboard", layout="wide")
st.title(":shopping_bags: Smart Retail & Customer Intelligence")


@st.cache_data(ttl=60)
def load_stats() -> dict:
    return _get("/dashboard/stats")


stats = load_stats()

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", stats.get("total_customers", 0))
c2.metric("Total Visits", stats.get("total_visits", 0))
c3.metric("Unique Visitors", stats.get("unique_visitors", 0))
top = stats.get("top_customer")
c4.metric("Top Customer", f"{top['name']} ({top['visits']})" if top else "—")

st.divider()

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Sentiment Distribution")
    sent_counts = stats.get("sentiment_counts", {})
    if any(sent_counts.values()):
        fig = px.pie(
            names=list(sent_counts.keys()),
            values=list(sent_counts.values()),
            color_discrete_map={"positive": "#2ca02c", "neutral": "#ffbb33", "negative": "#d62728"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No sentiment data yet.")

    st.subheader("Product Predictions by Category")
    prod_counts = stats.get("product_counts", {})
    if prod_counts:
        fig = px.bar(x=list(prod_counts.keys()), y=list(prod_counts.values()))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No product predictions yet.")

with col_b:
    st.subheader("Sentiment Trend (last 7 days)")
    trend = stats.get("sentiment_trend", [])
    if trend:
        df = pd.DataFrame(trend)
        fig = px.line(df, x="date", y=["positive", "neutral", "negative"], markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trend data yet.")

    st.subheader("Chatbot Intent Distribution")
    chat_counts = stats.get("chat_counts", {})
    if chat_counts:
        fig = px.bar(x=list(chat_counts.keys()), y=list(chat_counts.values()), orientation="h")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No chatbot logs yet.")

st.divider()

# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Recent Visits")
    visits = stats.get("recent_visits", [])
    if visits:
        st.dataframe(pd.DataFrame(visits), use_container_width=True)
    else:
        st.info("No visits recorded.")

with col_d:
    st.subheader("Recent Chatbot Interactions")
    chats = stats.get("recent_chats", [])
    if chats:
        st.dataframe(pd.DataFrame(chats), use_container_width=True)
    else:
        st.info("No chatbot interactions yet.")

st.divider()

# --------------------------------------------------------------------------- #
# Live demo widgets
# --------------------------------------------------------------------------- #
st.subheader("Live ML Demo")

tab1, tab2, tab3 = st.tabs(["Sentiment", "Product Classifier", "Chatbot"])

with tab1:
    text = st.text_area("Write a review or feedback:", value="I love this dress, it fits perfectly!")
    if st.button("Analyze sentiment"):
        try:
            result = _post("/analyze-sentiment", {"text": text})
            st.success(f"Sentiment: **{result['sentiment']}** (confidence {result['confidence']})")
            st.json(result.get("probabilities", {}))
        except Exception as exc:
            st.error(f"API error: {exc}")

with tab2:
    upload = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png", "webp"])
    if upload and st.button("Classify product"):
        try:
            result = _post("/classify-product", {"image_base64": _image_to_base64(upload)})
            top = result["top_prediction"]
            st.success(f"Top prediction: **{top['label']}** ({top['category']}) — {top['confidence']:.2%}")
            st.dataframe(pd.DataFrame(result["all_predictions"]))
        except Exception as exc:
            st.error(f"API error: {exc}")

with tab3:
    question = st.text_input("Ask the FAQ bot:", value="What are your store hours?")
    if st.button("Ask"):
        try:
            result = _post("/chatbot", {"message": question})
            st.success(f"Bot: {result['reply']}")
            st.caption(f"Intent: {result['intent']} — confidence {result['confidence']}")
        except Exception as exc:
            st.error(f"API error: {exc}")

st.divider()
st.caption(f"Data source: {API_BASE} | Auto-refreshes every 60s")
