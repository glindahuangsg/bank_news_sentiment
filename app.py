import os
import random
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import feedparser
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Constants
# ============================================================
SENTIMENT_MODEL_NAME = "pmatorras/financial-sentiment-analysis"
TOPIC_MODEL_NAME = "leonas5555/finnews-topic-single-classify"

SENTIMENT_COLOR_MAP = {
    "Positive": "#2ecc71",
    "Neutral": "#95a5a6",
    "Negative": "#e74c3c",
}

SENTIMENT_NORMALIZE = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "label_0": "Negative",
    "label_1": "Neutral",
    "label_2": "Positive",
}

RSS_FEEDS = [
    ("Reuters Business", "https://feeds.reuters.com/reuters/businessNews"),
    ("Reuters Companies", "https://feeds.reuters.com/reuters/companyNews"),
    ("SEC Press Releases",
     "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=40&output=atom"),
    ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
    ("Atlanta Fed", "https://www.atlantafed.org/rss/"),
]


# ============================================================
# Model loading (cached)
# ============================================================
@st.cache_resource(show_spinner="Loading financial sentiment model...")
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading topic classifier...")
def load_topic_model():
    tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(TOPIC_MODEL_NAME)
    model.eval()
    return tokenizer, model


# ============================================================
# Inference
# ============================================================
def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()
    raw_label = model.config.id2label.get(pred_id, "neutral")
    label = SENTIMENT_NORMALIZE.get(str(raw_label).lower(), "Neutral")
    return label, round(confidence, 2)


def predict_topic(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred_id].item()
    label = model.config.id2label.get(pred_id, f"Topic_{pred_id}")
    return label, round(confidence, 2)


# ============================================================
# Real news via RSS
# ============================================================
@st.cache_data(ttl=300, show_spinner="Fetching live news from RSS feeds...")
def fetch_real_news():
    rows = []
    news_id = 0
    failed_sources = []

    for source_name, rss_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                failed_sources.append(source_name)
                continue

            for entry in feed.entries[:15]:
                news_id += 1
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                published = entry.get("published", "")
                summary = entry.get("summary", "")
                # 去掉 HTML 标签（简单处理）
                summary = summary.replace("<p>", "").replace("</p>", "")
                summary = summary.replace("<br>", " ").replace("<br/>", " ")
                summary = summary[:400]

                try:
                    ts = parsedate_to_datetime(published)
                    if ts.tzinfo:
                        ts = ts.replace(tzinfo=None)
                except Exception:
                    ts = datetime.now()

                if not title:
                    continue

                rows.append({
                    "id": news_id,
                    "title": title,
                    "source": source_name,
                    "timestamp": ts,
                    "entity": "Unknown",
                    "snippet": summary if summary else title,
                    "link": link,
                })
        except Exception:
            failed_sources.append(source_name)
            continue

    if failed_sources:
        st.sidebar.warning(f"Failed RSS sources: {', '.join(failed_sources)}")

    if not rows:
        st.error("All RSS feeds failed. Please check network or feed URLs.")
        return pd.DataFrame(columns=[
            "id", "title", "source", "timestamp", "entity", "snippet", "link"
        ])

    df = pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


# ============================================================
# Enrich news with model predictions
# ============================================================
def enrich_news_with_models(df, sent_tok, sent_model, topic_tok, topic_model):
    if df.empty:
        return df

    sentiments, topics, sent_confs, topic_confs = [], [], [], []

    progress = st.progress(0, text="Running deep learning inference on news...")
    total = len(df)

    for i, text in enumerate(df["title"]):
        try:
            s_label, s_conf = predict_sentiment(text, sent_tok, sent_model)
        except Exception:
            s_label, s_conf = "Neutral", 0.0

        try:
            t_label, t_conf = predict_topic(text, topic_tok, topic_model)
        except Exception:
            t_label, t_conf = "Other", 0.0

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
# Alerts
# ============================================================
def generate_alerts(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "id", "severity", "sentiment", "topic", "title", "source",
            "timestamp", "confidence", "entity", "status"
        ])

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
        index=3,
    )

    topic_filter = st.sidebar.multiselect(
        "Topic",
        options=sorted(news_df["topic"].unique()) if not news_df.empty else [],
        default=sorted(news_df["topic"].unique()) if not news_df.empty else [],
    )

    sentiment_filter = st.sidebar.multiselect(
        "Sentiment",
        options=["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"],
    )

    entity_filter = st.sidebar.multiselect(
        "Entity",
        options=sorted(news_df["entity"].unique()) if not news_df.empty else [],
        default=sorted(news_df["entity"].unique()) if not news_df.empty else [],
    )

    keyword = st.sidebar.text_input("Keyword Search", "")

    return time_range, topic_filter, sentiment_filter, entity_filter, keyword


def apply_filters(df, time_range, topic_filter, sentiment_filter, entity_filter, keyword):
    if df.empty:
        return df

    filtered = df.copy()
    now = datetime.now()

    if time_range == "Last 1 hour":
        filtered = filtered[filtered["timestamp"] >= now - timedelta(hours=1)]
    elif time_range == "Last 6 hours":
        filtered = filtered[filtered["timestamp"] >= now - timedelta(hours=6)]
    elif time_range == "Last 24 hours":
        filtered = filtered[filtered["timestamp"] >= now - timedelta(hours=24)]

    if topic_filter:
        filtered = filtered[filtered["topic"].isin(topic_filter)]
    if sentiment_filter:
        filtered = filtered[filtered["sentiment"].isin(sentiment_filter)]
    if entity_filter:
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
               f"Live RSS + deep learning inference")

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
        if not filtered_news.empty:
            trend_df = filtered_news.copy()
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
            st.info("No data to display.")

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
            st.info("No data to display.")

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
        st.info("No news available.")


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
                if row.get("link"):
                    st.markdown(f"[Read full article]({row['link']})")
                st.markdown("**Model judgment basis:**")
                st.markdown(
                    f"- Sentiment: `{row['sentiment']}` "
                    f"(confidence {row['sentiment_confidence']})\n"
                    f"- Topic: `{row['topic']}` "
                    f"(confidence {row['topic_confidence']})\n"
                    f"- Entity: `{row['entity']}`\n"
                    f"- Model: `{SENTIMENT_MODEL_NAME}` / `{TOPIC_MODEL_NAME}`"
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
            return

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
                    sorted(news_df["topic"].unique()) if not news_df.empty else [],
                    default=sorted(news_df["topic"].unique())[:2] if not news_df.empty else [],
                )
            with c2:
                st.multiselect(
                    "Entity is",
                    sorted(news_df["entity"].unique()) if not news_df.empty else [],
                    default=sorted(news_df["entity"].unique())[:1] if not news_df.empty else [],
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
        neg = len(filtered_news[filtered_news["sentiment"] == "Negative"]) if total > 0 else 0
        neg_pct = (neg / total * 100) if total > 0 else 0
        high_alerts = len(alerts_df[alerts_df["severity"] == "High"]) if not alerts_df.empty else 0

        st.markdown(
            f"- Total news articles: **{total}**\n"
            f"- Negative articles: **{neg}** ({neg_pct:.1f}%)\n"
            f"- High priority alerts: **{high_alerts}**\n"
            f"- Model: `{SENTIMENT_MODEL_NAME}`"
        )

        st.subheader("2. Key Events")
        key_events = filtered_news[filtered_news["sentiment"] == "Negative"].head(5) if total > 0 else pd.DataFrame()
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

    # Login to Hugging Face if token is available in Streamlit secrets
    try:
        if "HF_TOKEN" in st.secrets:
            from huggingface_hub import login
            login(token=st.secrets["HF_TOKEN"])
    except Exception:
        pass

    # Load models
    sent_tok, sent_model = load_sentiment_model()
    topic_tok, topic_model = load_topic_model()

    # Fetch real news from RSS
    raw_news = fetch_real_news()

    if raw_news.empty:
        st.error("No news could be fetched from any RSS source. Please try again later.")
        st.stop()

    # Run inference (cached so it only runs once per session)
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
    st.sidebar.caption("Bank News Sentiment Analysis v0.4")
    st.sidebar.caption(f"Sentiment model: `{SENTIMENT_MODEL_NAME}`")
    st.sidebar.caption(f"Topic model: `{TOPIC_MODEL_NAME}`")
    st.sidebar.caption("Data source: Live RSS feeds")


if __name__ == "__main__":
    main()
