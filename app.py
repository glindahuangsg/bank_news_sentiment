import os

# Set HuggingFace cache to a writable dir BEFORE importing transformers
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# Constants
# ============================================================
SENTIMENT_MODEL_NAME = "pmatorras/financial-sentiment-multi-task"
TOPIC_MODEL_NAME = "leonas5555/finnews-topic-single-classify"

SENTIMENT_COLOR_MAP = {
    "Positive": "#2ecc71",
    "Neutral": "#95a5a6",
    "Negative": "#e74c3c",
}

TOPIC_LABELS = [
    "Analyst Ratings", "Banking", "Commodities", "Company Earnings",
    "Corporate Governance", "Cryptocurrency", "Economy", "Energy",
    "Financial Markets", "Healthcare", "Legal/Regulation", "M&A",
    "Real Estate", "Retail", "Technology", "Telecommunications",
    "Trade", "Transportation", "Utilities", "Other",
]

# Keyword rules for lightweight topic classification (no second model)
TOPIC_KEYWORDS = {
    "Analyst Ratings": ["analyst", "upgrade", "downgrade", "rating", "target price", "buy", "sell", "hold"],
    "Banking": ["bank", "banking", "lender", "loan", "deposit", "branch", "mortgage"],
    "Commodities": ["gold", "oil", "copper", "commodity", "commodities", "silver", "wheat"],
    "Company Earnings": ["earnings", "profit", "revenue", "net income", "eps", "quarter", "q1", "q2", "q3", "q4", "beats", "misses"],
    "Corporate Governance": ["board", "ceo", "cfo", "governance", "shareholder", "proxy", "director"],
    "Cryptocurrency": ["crypto", "bitcoin", "ethereum", "blockchain", "stablecoin", "token", "exchange"],
    "Economy": ["gdp", "inflation", "unemployment", "recession", "economic", "fed", "central bank", "rate cut", "rate hike"],
    "Energy": ["energy", "power", "solar", "wind", "renewable", "utility", "electricity"],
    "Financial Markets": ["market", "stocks", "equities", "bond", "yield", "index", "s&p", "nasdaq", "dow"],
    "Healthcare": ["health", "pharma", "drug", "hospital", "medical", "biotech", "fda"],
    "Legal/Regulation": ["regulator", "regulation", "fine", "penalty", "lawsuit", "court", "compliance", "aml", "investigation", "probe"],
    "M&A": ["merger", "acquisition", "acquire", "takeover", "deal", "buyout", "m&a"],
    "Real Estate": ["real estate", "property", "housing", "reit", "construction"],
    "Retail": ["retail", "consumer", "store", "e-commerce", "shopping", "sales"],
    "Technology": ["tech", "software", "ai", "cloud", "chip", "semiconductor", "digital"],
    "Telecommunications": ["telecom", "5g", "wireless", "broadband", "network"],
    "Trade": ["trade", "tariff", "export", "import", "supply chain", "sanction"],
    "Transportation": ["transport", "airline", "shipping", "logistics", "rail", "freight"],
    "Utilities": ["utility", "water", "gas", "electric", "pipeline"],
}


# ============================================================
# Model loading
# ============================================================
@st.cache_resource(show_spinner="Loading financial sentiment model...")
def load_sentiment_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load sentiment model: {e}")
        return None, None


@st.cache_resource(show_spinner="Loading topic classifier...")
def load_topic_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(TOPIC_MODEL_NAME)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"Topic model unavailable, using keyword fallback. ({e})")
        return None, None


# ============================================================
# Inference functions
# ============================================================
def predict_sentiment(text, tokenizer, model):
    if tokenizer is None or model is None:
        return "Neutral", 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
        id2label = model.config.id2label
        label = id2label.get(pred_id, "Neutral")
        label = label.capitalize()
        if label not in SENTIMENT_COLOR_MAP:
            label = "Neutral"
        return label, round(confidence, 2)
    except Exception:
        return "Neutral", 0.0


def predict_topic_keyword(text):
    """Lightweight keyword-based topic classifier (no model needed)."""
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    if not scores:
        return "Other", 0.3
    best_topic = max(scores, key=scores.get)
    total_hits = sum(scores.values())
    confidence = min(0.5 + 0.1 * scores[best_topic], 0.95)
    return best_topic, round(confidence, 2)


def predict_topic(text, tokenizer, model):
    """Use model if available, else fallback to keyword rules."""
    if tokenizer is None or model is None:
        return predict_topic_keyword(text)
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()
        id2label = model.config.id2label
        label = id2label.get(pred_id, f"Topic_{pred_id}")
        return label, round(confidence, 2)
    except Exception:
        return predict_topic_keyword(text)


# ============================================================
# Mock news generator
# ============================================================
@st.cache_data
def generate_mock_news(n=60):
    random.seed(42)
    np.random.seed(42)

    sources = ["Reuters", "Bloomberg", "Financial Times", "WSJ",
               "Central Bank Website", "Securities Daily", "Caixin",
               "Xinhua", "CNBC", "The Economist"]

    entities = ["Our Bank", "Competitor A", "Competitor B", "Regulator", "Key Client"]

    headlines = [
        "Central bank imposes fine on {entity} for AML violations",
        "New capital adequacy rules announced for commercial banks",
        "Regulator tightens scrutiny on wealth management products",
        "{entity} faces surge in customer complaints over wealth products",
        "Data breach reported at {entity} branch",
        "Loan default rate rises at {entity} in Q3",
        "Central bank cuts reserve requirement ratio by 25bps",
        "GDP growth beats expectations in Q3",
        "Inflation remains stable at 2.1%",
        "Banking sector sees consolidation amid digital transformation",
        "Fintech partnerships accelerate across major banks",
        "Green finance initiatives expand in banking sector",
        "{entity} reports net profit up 5% in Q3",
        "{entity} beats earnings estimates on strong loan growth",
        "{entity} net interest margin narrows slightly",
        "{entity} announces dividend increase",
        "{entity} launches new digital banking platform",
        "Analysts upgrade {entity} on improved outlook",
        "{entity} faces regulatory probe over mortgage practices",
        "Cybersecurity incident disrupts {entity} online services",
    ]

    now = datetime.now()
    rows = []
    for i in range(n):
        entity = random.choice(entities)
        template = random.choice(headlines)
        title = template.format(entity=entity)
        minutes_ago = random.randint(1, 1440)
        timestamp = now - timedelta(minutes=minutes_ago)

        rows.append({
            "id": i + 1,
            "title": title,
            "source": random.choice(sources),
            "timestamp": timestamp,
            "entity": entity,
            "snippet": f"{title}. Full article content would appear here in production. "
                       f"This is simulated data for local deployment testing.",
        })

    df = pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# Enrich news with model predictions
# ============================================================
def enrich_news_with_models(df, sent_tok, sent_model, topic_tok, topic_model):
    sentiments, topics, sent_confs, topic_confs = [], [], [], []

    progress = st.progress(0, text="Running inference on news...")
    total = len(df)

    for i, text in enumerate(df["title"]):
        s_label, s_conf = predict_sentiment(text, sent_tok, sent_model)
        t_label, t_conf = predict_topic(text, topic_tok, topic_model)

        sentiments.append(s_label)
        topics.append(t_label)
        sent_confs.append(s_conf)
        topic_confs.append(t_conf)

        progress.progress((i + 1) / total, text=f"Analyzing {i + 1}/{total}...")

    progress.empty()

    df = df.copy()
    df["sentiment"] = sentiments
    df["topic"] = topics
    df["sentiment_confidence"] = sent_confs
    df["topic_confidence"] = topic_confs
    return df


# ============================================================
# Alert generation
# ============================================================
def generate_alerts(df):
    alerts = []
    negative_df = df[df["sentiment"] == "Negative"].head(10)
    for _, row in negative_df.iterrows():
        alerts.append({
            "id": row["id"],
            "severity": "High" if row["sentiment_confidence"] > 0.85 else "Medium",
            "sentiment": row["sentiment"],
            "topic": row["topic"],
            "title": row["title"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "confidence": row["sentiment_confidence"],
            "entity": row["entity"],
            "status": random.choice(["Unresolved", "Resolved", "Ignored"]),
        })
    if not alerts:
        return pd.DataFrame(columns=[
            "id", "severity", "sentiment", "topic", "title", "source",
            "timestamp", "confidence", "entity", "status"
        ])
    return pd.DataFrame(alerts).sort_values("timestamp", ascending=False).reset_index(drop=True)


# ============================================================
# Sidebar filters
# ============================================================
def render_sidebar_filters(news_df):
    st.sidebar.title("🔎 Filters")

    time_range = st.sidebar.radio(
        "Time Range",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "All"],
        index=2,
    )

    topic_filter = st.sidebar.multiselect(
        "Topic",
        options=sorted(news_df["topic"].unique()),
        default=sorted(news_df["topic"].unique()),
    )

    sentiment_filter = st.sidebar.multiselect(
        "Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    entity_filter = st.sidebar.multiselect(
        "Entity",
        options=sorted(news_df["entity"].unique()),
        default=sorted(news_df["entity"].unique()),
    )

    keyword = st.sidebar.text_input("Keyword Search", "")

    return time_range, topic_filter, sentiment_filter, entity_filter, keyword


def apply_filters(df, time_range, topic_filter, sentiment_filter, entity_filter, keyword):
    filtered = df.copy()
    now = datetime.now()

    if time_range == "Last 1 hour":
        filtered = filtered[filtered["timestamp"] >= now - timedelta(hours=1)]
    elif time_range == "Last 6 hours":
        filtered = filtered[filtered["timestamp"] >= now - timedelta(hours=6)]
    elif time_range == "Last 24 hours":
        filtered = filtered[filtered["timestamp"] >= now - timedelta(hours=24)]

    filtered = filtered[filtered["topic"].isin(topic_filter)]
    filtered = filtered[filtered["sentiment"].isin(sentiment_filter)]
    filtered = filtered[filtered["entity"].isin(entity_filter)]

    if keyword:
        filtered = filtered[
            filtered["title"].str.contains(keyword, case=False, na=False) |
            filtered["snippet"].str.contains(keyword, case=False, na=False)
        ]

    return filtered


# ============================================================
# Page: Dashboard
# ============================================================
def render_dashboard(filtered_news, alerts_df):
    st.title("🏦 Bank News Sentiment Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
               f"Deep learning inference via Hugging Face")

    total_news = len(filtered_news)
    neg_count = len(filtered_news[filtered_news["sentiment"] == "Negative"])
    neg_ratio = (neg_count / total_news * 100) if total_news > 0 else 0
    high_alerts = len(alerts_df[alerts_df["severity"] == "High"]) if not alerts_df.empty else 0

    sentiment_score_map = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}
    avg_sentiment = (
        filtered_news["sentiment"].map(sentiment_score_map).mean()
        if total_news > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total News", f"{total_news:,}")
    with col2:
        st.metric("Negative Ratio", f"{neg_ratio:.1f}%",
                  delta="High" if neg_ratio > 25 else "Normal",
                  delta_color="inverse")
    with col3:
        st.metric("High Priority Alerts", high_alerts,
                  delta="Attention" if high_alerts > 0 else "Clear",
                  delta_color="inverse")
    with col4:
        st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")

    st.divider()

    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.subheader("Sentiment Trend (Last 24 Hours)")
        trend_df = filtered_news.copy()
        if not trend_df.empty:
            trend_df["hour"] = trend_df["timestamp"].dt.floor("h")
            trend_pivot = trend_df.groupby(["hour", "sentiment"]).size().unstack(fill_value=0)

            fig_trend = go.Figure()
            for sentiment in ["Positive", "Neutral", "Negative"]:
                if sentiment in trend_pivot.columns:
                    fig_trend.add_trace(go.Scatter(
                        x=trend_pivot.index,
                        y=trend_pivot[sentiment],
                        mode="lines+markers",
                        name=sentiment,
                        line=dict(color=SENTIMENT_COLOR_MAP[sentiment], width=2),
                    ))
            fig_trend.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No data in current filter.")

    with chart_col2:
        st.subheader("Sentiment Distribution")
        if not filtered_news.empty:
            dist = filtered_news["sentiment"].value_counts().reset_index()
            dist.columns = ["Sentiment", "Count"]
            fig_pie = px.pie(
                dist, names="Sentiment", values="Count",
                color="Sentiment",
                color_discrete_map=SENTIMENT_COLOR_MAP,
                hole=0.45,
            )
            fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data in current filter.")

    st.divider()

    st.subheader("🚨 High Priority Alerts")
    if alerts_df.empty:
        st.info("No alerts at the moment.")
    else:
        high_df = alerts_df[alerts_df["severity"] == "High"].head(5)
        if high_df.empty:
            st.info("No high priority alerts.")
        else:
            for _, row in high_df.iterrows():
                with st.container(border=True):
                    st.markdown(
                        f"<span style='color:#e74c3c; font-weight:bold;'>"
                        f"● {row['sentiment']}</span> | {row['topic']} | "
                        f"{row['timestamp'].strftime('%H:%M')}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{row['title']}**")
                    st.caption(
                        f"Source: {row['source']} | Confidence: {row['confidence']} | "
                        f"Entity: {row['entity']}"
                    )

    st.divider()

    st.subheader("📰 Latest News")
    if not filtered_news.empty:
        display_df = filtered_news.head(15)[
            ["title", "source", "timestamp", "sentiment", "topic", "sentiment_confidence"]
        ].copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No news matches current filters.")


# ============================================================
# Page: News Search
# ============================================================
def render_news_search(filtered_news):
    st.title("🔍 News Search & Filter")
    st.caption(f"{len(filtered_news)} articles match your filters")

    if filtered_news.empty:
        st.warning("No articles match the current filters. Please adjust the sidebar.")
        return

    for _, row in filtered_news.head(30).iterrows():
        color = SENTIMENT_COLOR_MAP.get(row["sentiment"], "#95a5a6")
        with st.container(border=True):
            st.markdown(f"### {row['title']}")
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>"
                f"{row['sentiment']}</span> | {row['topic']} | "
                f"Source: {row['source']} | {row['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Sentiment confidence: {row['sentiment_confidence']} | "
                f"Topic confidence: {row['topic_confidence']} | Entity: {row['entity']}"
            )
            with st.expander("View snippet & model analysis"):
                st.write(row["snippet"])
                st.markdown("**Model judgment basis:**")
                st.markdown(
                    f"- Sentiment: `{row['sentiment']}` "
                    f"(confidence {row['sentiment_confidence']})\n"
                    f"- Topic: `{row['topic']}` "
                    f"(confidence {row['topic_confidence']})\n"
                    f"- Entity: `{row['entity']}`\n"
                    f"- Model: `{SENTIMENT_MODEL_NAME}`"
                )


# ============================================================
# Page: Alert Center
# ============================================================
def render_alert_center(alerts_df, news_df):
    st.title("🚨 Alert Center")

    tab1, tab2 = st.tabs(["Alert History", "Rule Configuration"])

    with tab1:
        st.subheader("Alert History")
        if alerts_df.empty:
            st.info("No alerts generated from current data.")
        else:
            status_filter = st.multiselect(
                "Filter by status",
                options=["Unresolved", "Resolved", "Ignored"],
                default=["Unresolved", "Resolved", "Ignored"],
            )
            display_alerts = alerts_df[alerts_df["status"].isin(status_filter)]

            for _, row in display_alerts.iterrows():
                severity_color = "#e74c3c" if row["severity"] == "High" else "#e67e22"
                with st.container(border=True):
                    st.markdown(
                        f"<span style='color:{severity_color}; font-weight:bold;'>"
                        f"● {row['severity']} Priority</span> | {row['sentiment']} | "
                        f"{row['topic']} | {row['timestamp'].strftime('%Y-%m-%d %H:%M')}",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**{row['title']}**")
                    st.caption(
                        f"Source: {row['source']} | Confidence: {row['confidence']} | "
                        f"Entity: {row['entity']} | Status: {row['status']}"
                    )

    with tab2:
        st.subheader("Create Alert Rule")
        with st.form("alert_rule_form"):
            rule_name = st.text_input("Rule Name", "Regulatory Penalty Alert")

            st.markdown("**Trigger Conditions (all must be met):**")
            c1, c2 = st.columns(2)
            with c1:
                st.multiselect(
                    "Sentiment is",
                    ["Positive", "Neutral", "Negative"],
                    default=["Negative"],
                )
                st.multiselect(
                    "Topic is",
                    sorted(news_df["topic"].unique()),
                    default=sorted(news_df["topic"].unique())[:2],
                )
            with c2:
                st.multiselect(
                    "Entity is",
                    sorted(news_df["entity"].unique()),
                    default=["Our Bank"],
                )
                st.number_input(
                    "Volume threshold (similar reports within 1 hour)",
                    min_value=1, max_value=50, value=3,
                )

            st.markdown("**Notification Channels:**")
            n1, n2, n3 = st.columns(3)
            n1.checkbox("In-app", value=True)
            n2.checkbox("Email", value=True)
            n3.checkbox("SMS", value=False)

            submitted = st.form_submit_button("Save Rule")
            if submitted:
                st.success(f"Rule '{rule_name}' saved successfully. (Simulated)")


# ============================================================
# Page: Report Generator
# ============================================================
def render_report_generator(filtered_news, alerts_df):
    st.title("📄 Daily Sentiment Report")

    report_date = st.date_input("Report Date", datetime.now().date())
    report_type = st.radio("Report Type", ["Daily", "Weekly"], horizontal=True)

    if st.button("Generate Report", type="primary"):
        st.success("Report generated successfully.")

        st.divider()
        st.header(f"Bank News Sentiment Report — {report_type}")
        st.caption(f"Report Date: {report_date}")

        st.subheader("1. Overview")
        total = len(filtered_news)
        neg = len(filtered_news[filtered_news["sentiment"] == "Negative"])
        neg_pct = (neg / total * 100) if total > 0 else 0
        high_alerts = len(alerts_df[alerts_df["severity"] == "High"]) if not alerts_df.empty else 0

        st.markdown(
            f"- Total news articles: **{total}**\n"
            f"- Negative articles: **{neg}** ({neg_pct:.1f}%)\n"
            f"- High priority alerts: **{high_alerts}**\n"
            f"- Model: `{SENTIMENT_MODEL_NAME}`"
        )

        st.subheader("2. Key Events")
        key_events = filtered_news[filtered_news["sentiment"] == "Negative"].head(5)
        if key_events.empty:
            st.info("No negative events in current filter.")
        else:
            for i, (_, row) in enumerate(key_events.iterrows(), 1):
                st.markdown(
                    f"{i}. **{row['title']}** "
                    f"({row['sentiment']}, {row['topic']}) — "
                    f"{row['source']}, {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"
                )

        st.subheader("3. Entity Sentiment Profile")
        if not filtered_news.empty:
            entity_profile = filtered_news.groupby(
                ["entity", "sentiment"]
            ).size().unstack(fill_value=0)
            st.dataframe(entity_profile, use_container_width=True)

        st.subheader("4. Suggested Focus Areas")
        st.markdown(
            "- Monitor negative sentiment spikes in regulatory topics\n"
            "- Investigate entities with rising negative coverage\n"
            "- Track topic distribution shifts over time"
        )

        st.download_button(
            "Download Report as Markdown",
            data=f"# Bank News Sentiment Report\n\nGenerated: {datetime.now()}\n",
            file_name=f"sentiment_report_{report_date}.md",
            mime="text/markdown",
        )


# ============================================================
# Main
# ============================================================
def main():
    st.set_page_config(
        page_title="Bank News Sentiment Analysis",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load sentiment model (required)
    sent_tok, sent_model = load_sentiment_model()

    # Topic model toggle (optional, saves RAM on free tier)
    st.sidebar.markdown("---")
    use_topic_model = st.sidebar.checkbox(
        "Load topic model (uses more RAM)",
        value=False,
        help="Uncheck to use fast keyword-based topic classification instead.",
    )

    if use_topic_model:
        topic_tok, topic_model = load_topic_model()
    else:
        topic_tok, topic_model = None, None
        st.sidebar.caption("Using keyword-based topic classifier.")

    # Generate mock news
    raw_news = generate_mock_news(60)

    # Run inference (cached)
    @st.cache_data(show_spinner=False)
    def get_enriched_news(_sent_tok, _sent_model, _topic_tok, _topic_model, raw_df):
        return enrich_news_with_models(
            raw_df, _sent_tok, _sent_model, _topic_tok, _topic_model
        )

    news_df = get_enriched_news(sent_tok, sent_model, topic_tok, topic_model, raw_news)
    alerts_df = generate_alerts(news_df)

    # Sidebar filters
    time_range, topic_filter, sentiment_filter, entity_filter, keyword = render_sidebar_filters(news_df)

    # Navigation
    page = st.sidebar.radio(
        "📄 Page",
        ["Dashboard", "News Search", "Alert Center", "Report Generator"],
    )

    # Apply filters
    filtered_news = apply_filters(
        news_df, time_range, topic_filter, sentiment_filter, entity_filter, keyword
    )

    # Route to page
    if page == "Dashboard":
        render_dashboard(filtered_news, alerts_df)
    elif page == "News Search":
        render_news_search(filtered_news)
    elif page == "Alert Center":
        render_alert_center(alerts_df, news_df)
    elif page == "Report Generator":
        render_report_generator(filtered_news, alerts_df)

    # Footer
    st.sidebar.divider()
    st.sidebar.caption("Bank News Sentiment Analysis v0.3")
    st.sidebar.caption(f"Model: `{SENTIMENT_MODEL_NAME}`")


if __name__ == "__main__":
    main()
