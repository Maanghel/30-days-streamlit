# Import relevant libraries
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def style_negative(value, props=""):
    """Style negative numeric values."""
    try:
        return props if value < 0 else None
    except (TypeError, ValueError):
        return None


def style_positive(value, props=""):
    """Style positive numeric values."""
    try:
        return props if value > 0 else None
    except (TypeError, ValueError):
        return None


def audience_simple(country):
    """Group countries into the main audience categories."""
    if country == "US":
        return "USA"
    if country == "IN":
        return "India"
    return "Other"


def parse_dates(series):
    """Parse mixed date formats, including the nonstandard 'Sept' spelling."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\bSept\b", "Sep", regex=True)
    )
    return pd.to_datetime(
        cleaned,
        format="mixed",
        dayfirst=True,
        errors="coerce",
    )


def clean_numeric_columns(df, columns):
    """Convert selected columns to numeric values, coercing invalid entries."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the CSV files and perform basic cleaning and feature engineering."""
    base_path = "day4"

    df_agg = pd.read_csv(
        f"{base_path}/Aggregated_Metrics_By_Video.csv"
    ).iloc[1:, :].copy()

    expected_columns = [
        "Video",
        "Video title",
        "Video publish time",
        "Comments added",
        "Shares",
        "Dislikes",
        "Likes",
        "Subscribers lost",
        "Subscribers gained",
        "RPM(USD)",
        "CPM(USD)",
        "Average % viewed",
        "Average view duration",
        "Views",
        "Watch time (hours)",
        "Subscribers",
        "Your estimated revenue (USD)",
        "Impressions",
        "Impressions ctr(%)",
    ]

    if len(df_agg.columns) == len(expected_columns):
        df_agg.columns = expected_columns

    required_agg_columns = [
        "Video",
        "Video title",
        "Video publish time",
        "Comments added",
        "Shares",
        "Dislikes",
        "Likes",
        "Subscribers gained",
        "RPM(USD)",
        "Average % viewed",
        "Average view duration",
        "Views",
        "Subscribers",
    ]
    missing_agg = [c for c in required_agg_columns if c not in df_agg.columns]
    if missing_agg:
        raise ValueError(
            "Missing columns in Aggregated_Metrics_By_Video.csv: "
            + ", ".join(missing_agg)
        )

    df_agg["Video publish time"] = parse_dates(df_agg["Video publish time"])

    numeric_agg_columns = [
        "Comments added",
        "Shares",
        "Dislikes",
        "Likes",
        "Subscribers lost",
        "Subscribers gained",
        "RPM(USD)",
        "CPM(USD)",
        "Average % viewed",
        "Views",
        "Watch time (hours)",
        "Subscribers",
        "Your estimated revenue (USD)",
        "Impressions",
        "Impressions ctr(%)",
    ]
    df_agg = clean_numeric_columns(df_agg, numeric_agg_columns)

    duration = pd.to_timedelta(
        df_agg["Average view duration"].astype("string"),
        errors="coerce",
    )
    df_agg["Avg_duration_sec"] = duration.dt.total_seconds()

    engagement_numerator = (
        df_agg["Comments added"].fillna(0)
        + df_agg["Shares"].fillna(0)
        + df_agg["Dislikes"].fillna(0)
        + df_agg["Likes"].fillna(0)
    )
    views_denominator = df_agg["Views"].replace(0, np.nan)
    subs_denominator = df_agg["Subscribers gained"].replace(0, np.nan)

    df_agg["Engagement_ratio"] = engagement_numerator / views_denominator
    df_agg["Views / sub gained"] = df_agg["Views"] / subs_denominator
    df_agg.sort_values(
        "Video publish time",
        ascending=False,
        inplace=True,
        na_position="last",
    )

    df_agg_sub = pd.read_csv(
        f"{base_path}/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv"
    )
    df_time = pd.read_csv(
        f"{base_path}/Video_Performance_Over_Time.csv"
    )

    if "Date" not in df_time.columns:
        raise ValueError(
            "The Date column is missing from Video_Performance_Over_Time.csv."
        )

    original_dates = df_time["Date"].copy()
    df_time["Date"] = parse_dates(df_time["Date"])

    invalid_dates = original_dates[df_time["Date"].isna()]
    if not invalid_dates.empty:
        st.warning(
            f"{len(invalid_dates)} date value(s) in "
            "Video_Performance_Over_Time.csv could not be parsed. "
            "Those rows will be excluded from time-based charts."
        )

    if "Views" in df_time.columns:
        df_time["Views"] = pd.to_numeric(df_time["Views"], errors="coerce")

    return df_agg, df_agg_sub, df_time


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
try:
    df_agg, df_agg_sub, df_time = load_data()
except FileNotFoundError as error:
    st.error(
        f"Could not find a required CSV file: {error}. "
        "Run this app from the project root and verify the files are in the day4/ folder."
    )
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

if df_agg.empty:
    st.error("The video metrics CSV does not contain any video rows.")
    st.stop()

# Join daily performance to video publication dates.
if "External Video ID" not in df_time.columns:
    st.error(
        "The column 'External Video ID' is missing from "
        "Video_Performance_Over_Time.csv."
    )
    st.stop()

if "Video Title" not in df_time.columns:
    st.error(
        "The column 'Video Title' is missing from "
        "Video_Performance_Over_Time.csv."
    )
    st.stop()

df_time_diff = pd.merge(
    df_time,
    df_agg[["Video", "Video title", "Video publish time"]],
    left_on="External Video ID",
    right_on="Video",
    how="inner",
)
df_time_diff["days_published"] = (
    df_time_diff["Date"] - df_time_diff["Video publish time"]
).dt.days

# Use the latest publication date as the reference for the last 12 months.
latest_publish_date = df_agg["Video publish time"].max()
if pd.isna(latest_publish_date):
    st.error("No valid publication dates were found in the video metrics CSV.")
    st.stop()

date_12mo = latest_publish_date - pd.DateOffset(months=12)
df_time_diff_yr = df_time_diff[
    df_time_diff["Video publish time"] >= date_12mo
].copy()

# Build daily benchmark statistics for the first 30 days.
first_30_data = df_time_diff_yr[
    df_time_diff_yr["days_published"].between(0, 30)
].copy()

if not first_30_data.empty:
    views_days = (
        first_30_data.groupby("days_published")["Views"]
        .agg(
            mean_views="mean",
            median_views="median",
            pct_80=lambda values: values.quantile(0.80),
            pct_20=lambda values: values.quantile(0.20),
        )
        .reset_index()
        .sort_values("days_published")
    )

    # This is a benchmark made by summing each day's daily statistic.
    # It is not the exact percentile of each video's cumulative views.
    views_cumulative = views_days.copy()
    benchmark_columns = ["median_views", "pct_80", "pct_20"]
    views_cumulative[benchmark_columns] = (
        views_cumulative[benchmark_columns].cumsum()
    )
else:
    views_cumulative = pd.DataFrame(
        columns=["days_published", "median_views", "pct_80", "pct_20"]
    )


# ---------------------------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YouTube Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

analysis_mode = st.sidebar.selectbox(
    "Aggregate or Individual Video",
    ("Aggregate Metrics", "Individual Video Analysis"),
)


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------
if analysis_mode == "Aggregate Metrics":
    st.title("Ken Jee YouTube Aggregated Data")

    metric_columns = [
        "Video publish time",
        "Views",
        "Likes",
        "Subscribers",
        "Shares",
        "Comments added",
        "RPM(USD)",
        "Average % viewed",
        "Avg_duration_sec",
        "Engagement_ratio",
        "Views / sub gained",
    ]
    df_agg_metrics = df_agg[metric_columns].copy()

    metric_date_6mo = latest_publish_date - pd.DateOffset(months=6)
    metric_date_12mo = latest_publish_date - pd.DateOffset(months=12)

    metrics_6mo = df_agg_metrics[
        df_agg_metrics["Video publish time"] >= metric_date_6mo
    ].median(numeric_only=True)
    metrics_12mo = df_agg_metrics[
        df_agg_metrics["Video publish time"] >= metric_date_12mo
    ].median(numeric_only=True)

    st.subheader("Median metrics: last 6 months")
    metric_names = list(metrics_6mo.index)
    columns = st.columns(5)

    for index, metric_name in enumerate(metric_names[:5]):
        value_6mo = metrics_6mo.get(metric_name, np.nan)
        value_12mo = metrics_12mo.get(metric_name, np.nan)

        if pd.notna(value_12mo) and value_12mo != 0 and pd.notna(value_6mo):
            delta = (value_6mo - value_12mo) / abs(value_12mo)
            delta_text = f"{delta:.2%}"
        else:
            delta_text = None

        with columns[index]:
            st.metric(
                label=metric_name,
                value=f"{value_6mo:,.1f}" if pd.notna(value_6mo) else "N/A",
                delta=delta_text,
            )

    st.caption(
        "The delta compares the last 6 months with the last 12 months; "
        "the 6-month period is included in the 12-month period."
    )

    # Preserve original metric values for the table.
    df_table = df_agg.copy()
    df_table["Publish_date"] = df_table["Video publish time"].dt.date
    table_columns = [
        "Video title",
        "Publish_date",
        "Views",
        "Likes",
        "Subscribers",
        "Shares",
        "Comments added",
        "RPM(USD)",
        "Average % viewed",
        "Avg_duration_sec",
        "Engagement_ratio",
        "Views / sub gained",
    ]
    df_table = df_table[table_columns]

    st.subheader("Video metrics")
    st.dataframe(
        df_table.style
        .map(style_negative, props="color: red;")
        .map(style_positive, props="color: green;")
        .format(
            {
                "Views": "{:,.0f}",
                "Likes": "{:,.0f}",
                "Subscribers": "{:,.0f}",
                "Shares": "{:,.0f}",
                "Comments added": "{:,.0f}",
                "RPM(USD)": "${:,.2f}",
                "Average % viewed": "{:.1f}%",
                "Avg_duration_sec": "{:,.1f}",
                "Engagement_ratio": "{:.2%}",
                "Views / sub gained": "{:,.1f}",
            },
            na_rep="—",
        ),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Individual video analysis
# ---------------------------------------------------------------------------
else:
    st.title("Individual Video Performance")

    videos = df_agg["Video title"].dropna().astype(str).tolist()
    if not videos:
        st.warning("No video titles are available to select.")
        st.stop()

    selected_video = st.selectbox("Pick a Video:", videos)
    selected_rows = df_agg[df_agg["Video title"] == selected_video]

    if selected_rows.empty:
        st.warning("No aggregated data was found for the selected video.")
        st.stop()

    video_id = selected_rows.iloc[0]["Video"]

    # Audience by country and subscriber status.
    if {"Video Title", "Country Code", "Is Subscribed", "Views"}.issubset(
        df_agg_sub.columns
    ):
        audience = df_agg_sub[
            df_agg_sub["Video Title"] == selected_video
        ].copy()

        audience["Views"] = pd.to_numeric(
            audience["Views"], errors="coerce"
        )
        audience["Country"] = audience["Country Code"].apply(audience_simple)

        if not audience.empty:
            audience = (
                audience.groupby(["Is Subscribed", "Country"], as_index=False)["Views"]
                .sum()
            )
            fig_audience = px.bar(
                audience,
                x="Views",
                y="Is Subscribed",
                color="Country",
                orientation="h",
                barmode="stack",
                title="Views by subscriber status and country",
                labels={
                    "Views": "Views",
                    "Is Subscribed": "Subscriber status",
                    "Country": "Audience country",
                },
            )
            st.plotly_chart(fig_audience, use_container_width=True)
        else:
            st.info("No audience breakdown is available for this video.")
    else:
        st.warning(
            "The country/subscriber CSV does not contain all expected columns "
            "(Video Title, Country Code, Is Subscribed, Views)."
        )

    # Compare cumulative views for this video with the benchmark.
    current_video = df_time_diff[
        df_time_diff["External Video ID"] == video_id
    ].copy()
    current_video = current_video[
        current_video["days_published"].between(0, 30)
    ].sort_values("days_published")

    fig_views = go.Figure()

    if not views_cumulative.empty:
        fig_views.add_trace(
            go.Scatter(
                x=views_cumulative["days_published"],
                y=views_cumulative["pct_20"],
                mode="lines",
                name="20th percentile benchmark",
                line=dict(color="purple", dash="dash"),
            )
        )
        fig_views.add_trace(
            go.Scatter(
                x=views_cumulative["days_published"],
                y=views_cumulative["median_views"],
                mode="lines",
                name="Median benchmark",
                line=dict(color="black", dash="dash"),
            )
        )
        fig_views.add_trace(
            go.Scatter(
                x=views_cumulative["days_published"],
                y=views_cumulative["pct_80"],
                mode="lines",
                name="80th percentile benchmark",
                line=dict(color="royalblue", dash="dash"),
            )
        )

    if not current_video.empty:
        current_video["Cumulative views"] = current_video["Views"].cumsum()
        fig_views.add_trace(
            go.Scatter(
                x=current_video["days_published"],
                y=current_video["Cumulative views"],
                mode="lines+markers",
                name="Selected video",
                line=dict(color="firebrick", width=4),
            )
        )
    else:
        st.info(
            "No daily view history was found for this video during its first 30 days."
        )

    fig_views.update_layout(
        title="View comparison: first 30 days",
        xaxis_title="Days since publication",
        yaxis_title="Cumulative views (benchmark is approximate)",
        legend_title="Series",
    )
    st.plotly_chart(fig_views, use_container_width=True)