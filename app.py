import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Food Price Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("🍎 Food Price Inflation Analytics")
st.write("Food Price Analysis and Forecasting Dashboard")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# Sample dataset so the dashboard runs without an upload
@st.cache_data
def sample_data():
    dates = pd.date_range("2023-01-01", periods=36, freq="MS")
    foods = {
        "Rice": 40,
        "Wheat": 30,
        "Tomato": 25,
        "Potato": 20,
        "Onion": 28
    }
    rows = []
    for food, base in foods.items():
        for i, date in enumerate(dates):
            price = base + i * base * 0.006
            price += 2 * np.sin(i * 2 * np.pi / 12)
            rows.append({
                "Date": date,
                "Food": food,
                "Price": round(max(1, price), 2)
            })
    return pd.DataFrame(rows)

try:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = sample_data()
        st.info(
            "Sample data is being shown. "
            "Upload your own CSV to analyze your dataset."
        )

    # Match common column names
    aliases = {
        "date": "Date",
        "month": "Date",
        "food": "Food",
        "product": "Food",
        "item": "Food",
        "commodity": "Food",
        "price": "Price",
        "modal_price": "Price",
        "average_price": "Price"
    }

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in aliases:
            rename[col] = aliases[key]

    df = df.rename(columns=rename)

    required = ["Date", "Food", "Price"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        st.error(
            "Missing columns: " + ", ".join(missing) +
            ". Your CSV needs Date, Food and Price columns."
        )
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Food"] = df["Food"].astype("string").str.strip()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Date", "Food", "Price"])
    df = df[(df["Food"] != "") & (df["Price"] > 0)]
    df = df.sort_values("Date")

    if df.empty:
        st.error("No valid data found in the CSV.")
        st.stop()

    # Filters
    st.sidebar.header("Filters")
    foods = sorted(df["Food"].unique().tolist())
    selected_foods = st.sidebar.multiselect(
        "Select food products",
        foods,
        default=foods
    )

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    filtered = df[df["Food"].isin(selected_foods)].copy()

    if len(date_range) == 2:
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        filtered = filtered[
            (filtered["Date"] >= start) &
            (filtered["Date"] < end)
        ]

    if filtered.empty:
        st.warning("No records match your filters.")
        st.stop()

    # Monthly average for consistent comparisons
    monthly = (
        filtered.assign(
            Month=filtered["Date"].dt.to_period("M").dt.to_timestamp()
        )
        .groupby(["Month", "Food"], as_index=False)["Price"]
        .mean()
        .sort_values("Month")
    )

    st.subheader("📌 Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(filtered):,}")
    c2.metric("Average Price", f"{filtered['Price'].mean():.2f}")
    c3.metric("Minimum Price", f"{filtered['Price'].min():.2f}")
    c4.metric("Maximum Price", f"{filtered['Price'].max():.2f}")

    st.caption(
        "Prices use the units present in your dataset. "
        "Sample data is illustrative, not real market data."
    )

    # Price trend
    st.subheader("📈 Food Price Trends")
    fig = px.line(
        monthly,
        x="Month",
        y="Price",
        color="Food",
        markers=True,
        title="Monthly Average Price by Food Product"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Product comparison
    st.subheader("📊 Average Price Comparison")
    comparison = (
        filtered.groupby("Food", as_index=False)["Price"]
        .mean()
        .sort_values("Price", ascending=False)
    )
    fig2 = px.bar(
        comparison,
        x="Food",
        y="Price",
        color="Food",
        title="Average Price of Food Products"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Inflation calculation: first vs last monthly average
    st.subheader("📉 Price Change Analysis")
    changes = []
    for food, group in monthly.groupby("Food"):
        group = group.sort_values("Month")
        first = group.iloc[0]["Price"]
        last = group.iloc[-1]["Price"]
        pct = ((last - first) / first * 100) if first != 0 else np.nan
        changes.append({
            "Food": food,
            "First Monthly Price": round(first, 2),
            "Latest Monthly Price": round(last, 2),
            "Price Change (%)": round(pct, 2)
        })

    change_df = pd.DataFrame(changes)
    st.dataframe(change_df, use_container_width=True)

    fig3 = px.bar(
        change_df,
        x="Food",
        y="Price Change (%)",
        color="Price Change (%)",
        title="Price Change Over Selected Period"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Forecasting
    st.subheader("🔮 Future Price Forecasting")

    available = sorted(monthly["Food"].unique().tolist())
    chosen = st.selectbox("Choose a food product", available)
    periods = st.slider(
        "Months to forecast",
        min_value=1,
        max_value=12,
        value=6
    )

    series = (
        monthly[monthly["Food"] == chosen]
        .sort_values("Month")
        .dropna(subset=["Price"])
        .copy()
    )

    if len(series) < 3:
        st.warning("At least 3 monthly observations are needed.")
    else:
        series["t"] = np.arange(len(series))
        model = LinearRegression()
        model.fit(series[["t"]], series["Price"])

        future_t = np.arange(
            len(series), len(series) + periods
        )
        predictions = model.predict(
            pd.DataFrame({"t": future_t})
        )
        future_dates = pd.date_range(
            series["Month"].max() + pd.offsets.MonthBegin(1),
            periods=periods,
            freq="MS"
        )

        forecast = pd.DataFrame({
            "Month": future_dates,
            "Price": np.maximum(predictions, 0).round(2),
            "Type": "Forecast"
        })

        historical = series[["Month", "Price"]].rename(
            columns={"Month": "Month"}
        ).copy()
        historical["Type"] = "Historical"

        chart_data = pd.concat([
            historical, forecast
        ], ignore_index=True)

        fig4 = px.line(
            chart_data,
            x="Month",
            y="Price",
            color="Type",
            markers=True,
            title=f"{chosen}: Historical and Forecast Prices"
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.dataframe(forecast, use_container_width=True)

        st.caption(
            "Forecasts are estimates from a simple linear regression "
            "trend, not guaranteed market prices. Accuracy depends "
            "on data quality and real-world factors."
        )

    # Download
    st.subheader("⬇️ Download Results")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered Dataset",
        data=csv,
        file_name="food_price_analysis.csv",
        mime="text/csv"
    )

    st.subheader("📄 Dataset Preview")
    st.dataframe(filtered.head(100), use_container_width=True)

except Exception as e:
    st.error(f"Could not process the dataset: {e}")
    st.info(
        "Check that the CSV is valid and contains Date, Food and Price columns."
    )