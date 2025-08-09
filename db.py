# Streamlit Dashboard for Your Jupyter Notebook Analyses
# -----------------------------------------------------
# This single-file app follows the tutorial structure from your slides:
# - Text & metrics
# - Visualisations (Matplotlib / Altair / Plotly)
# - Interactivity (sidebar controls, widgets)
# - File upload so you can plug in CSVs exported from your notebook
#
# How to run:
# 1) pip install -U streamlit pandas numpy altair plotly matplotlib
# 2) Save this file as app.py (or keep the current name) in a folder with your CSVs (optional).
# 3) streamlit run app.py
#
# Expected CSV schemas (case-insensitive for column names):
#  A) Emissions (e.g., CO2): Country, Year, Value
#  B) Temperature (annual mean): Country, Year, Value
#  C) GDP Growth: Country, Year, gdp_growth  (or Value)
#  D) Disasters (yearly counts): Year, <one column per disaster type>
#
# Tips:
# - If you don't upload data, the app loads small example datasets so you can see how it works.
# - Use the sidebar to filter Country and Year range, smoothing window, and log scale.
# - Tabs group your analyses: Overview, Emissions, Temperature, GDP, Disasters.

from __future__ import annotations
import io
import textwrap
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
from matplotlib import pyplot as plt

st.set_page_config(page_title="Data Dashboard", page_icon="📊", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case and strip column names for flexible matching."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_csv(upload: Optional[st.runtime.uploaded_file_manager.UploadedFile],
             sample_loader) -> pd.DataFrame:
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        df = sample_loader()
    return df

# Sample data so the app works out-of-the-box

def sample_emissions() -> pd.DataFrame:
    prng = np.random.default_rng(78557)
    years = np.arange(1990, 2021)
    countries = ["Germany", "France", "United States", "Japan"]
    rows = []
    for c in countries:
        base = prng.uniform(6, 12)
        trend = prng.uniform(-0.05, 0.05)
        for y in years:
            val = base + trend * (y - years[0]) + prng.normal(0, 0.3)
            rows.append((c, y, max(val, 0.1)))
    return pd.DataFrame(rows, columns=["Country", "Year", "Value"])


def sample_temperature() -> pd.DataFrame:
    prng = np.random.default_rng(42)
    years = np.arange(1990, 2021)
    countries = ["Germany", "France", "United States", "Japan"]
    rows = []
    for c in countries:
        base = prng.uniform(7, 16)
        trend = prng.uniform(0.01, 0.05)
        for y in years:
            val = base + trend * (y - 1990) + prng.normal(0, 0.2)
            rows.append((c, y, val))
    return pd.DataFrame(rows, columns=["Country", "Year", "Value"])


def sample_gdp() -> pd.DataFrame:
    prng = np.random.default_rng(7)
    years = np.arange(1990, 2021)
    countries = ["Germany", "France", "United States", "Japan"]
    rows = []
    for c in countries:
        for y in years:
            growth = prng.normal(2.0, 1.5)
            rows.append((c, y, growth))
    return pd.DataFrame(rows, columns=["Country", "Year", "gdp_growth"])


def sample_disasters() -> pd.DataFrame:
    prng = np.random.default_rng(9)
    years = np.arange(1990, 2021)
    types = ["Earthquake", "Epidemic", "Extreme temperature", "Flood", "Mass movement (wet)", "Storm"]
    data = {"Year": years}
    for t in types:
        data[t] = prng.poisson(lam=np.linspace(1, 5, len(years)))
    return pd.DataFrame(data)


def year_range(df: pd.DataFrame, year_col: str = "year") -> Tuple[int, int]:
    ymin, ymax = int(df[year_col].min()), int(df[year_col].max())
    return ymin, ymax


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------
# Sidebar: Data Inputs & Controls
# -----------------------------

st.sidebar.title("Controls")

st.sidebar.markdown("**Upload your CSVs** (optional). If empty, demo data will be used.")
up_em = st.sidebar.file_uploader("Emissions CSV", type=["csv"], key="em_up")
up_tmp = st.sidebar.file_uploader("Temperature CSV", type=["csv"], key="tp_up")
up_gdp = st.sidebar.file_uploader("GDP Growth CSV", type=["csv"], key="gd_up")
up_dis = st.sidebar.file_uploader("Disasters CSV", type=["csv"], key="ds_up")

em = _norm_cols(load_csv(up_em, sample_emissions))
tp = _norm_cols(load_csv(up_tmp, sample_temperature))
gx = _norm_cols(load_csv(up_gdp, sample_gdp))
ds = _norm_cols(load_csv(up_dis, sample_disasters))

# Standardize columns
if "value" not in em.columns:
    # allow 'emissions' naming
    if "emissions" in em.columns:
        em = em.rename(columns={"emissions": "value"})
if "gdp_growth" not in gx.columns and "value" in gx.columns:
    gx = gx.rename(columns={"value": "gdp_growth"})

# Country & year filters (applied where relevant)
common_countries = sorted(set(em.get("country", pd.Series(dtype=str))).intersection(
    set(tp.get("country", pd.Series(dtype=str))).intersection(
        set(gx.get("country", pd.Series(dtype=str)))
)))

default_country = common_countries[0] if common_countries else None
country = st.sidebar.selectbox("Country", options=common_countries or ["(no country)"] , index=0)

# Year range based on emissions first, fallback to others
for_base = None
for dfcand in [em, tp, gx, ds]:
    if "year" in dfcand.columns and len(dfcand):
        for_base = dfcand
        break

if for_base is not None and len(for_base):
    ymin, ymax = year_range(for_base)
else:
    ymin, ymax = 1990, 2020

yr = st.sidebar.slider("Year range", min_value=int(ymin), max_value=int(ymax), value=(int(ymin), int(ymax)))

smooth = st.sidebar.slider("Moving average window (years)", 1, 9, 1, 2)
log_scale = st.sidebar.checkbox("Log scale for emissions", value=False)

st.sidebar.divider()
show_raw = st.sidebar.checkbox("Show raw data tables", value=False)

# -----------------------------
# Header
# -----------------------------

st.title("📊 Interactive Data Dashboard")
st.caption("Built with Streamlit · Text, metrics, visualisations, and interactivity")

# -----------------------------
# Tabs
# -----------------------------

tab_overview, tab_emissions, tab_temp, tab_gdp, tab_dis = st.tabs([
    "Overview", "Emissions", "Temperature", "GDP Growth", "Disasters"
])

# -----------------------------
# Overview Tab
# -----------------------------
with tab_overview:
    st.subheader("Key Metrics")

    # Compute some metrics for the chosen country and range
    def _subset_by_country_year(df: pd.DataFrame, value_col: str) -> Tuple[Optional[float], Optional[float]]:
        if df.empty or "year" not in df.columns:
            return None, None
        sub = df.copy()
        if "country" in sub.columns:
            sub = sub[sub["country"].str.lower() == str(country).lower()]
        sub = sub[(sub["year"] >= yr[0]) & (sub["year"] <= yr[1])]
        if sub.empty or value_col not in sub.columns:
            return None, None
        sub = ensure_numeric(sub, [value_col])
        sub = sub.dropna(subset=[value_col])
        if sub.empty:
            return None, None
        first = sub.sort_values("year").iloc[0][value_col]
        last = sub.sort_values("year").iloc[-1][value_col]
        delta = last - first
        return last, delta

    m1, d1 = _subset_by_country_year(em, "value")
    m2, d2 = _subset_by_country_year(tp, "value")
    m3, d3 = _subset_by_country_year(gx, "gdp_growth")

    c1, c2, c3 = st.columns(3)
    c1.metric("Emissions (last in range)", f"{m1:.2f}" if m1 is not None else "–", f"{d1:+.2f}" if d1 is not None else None)
    c2.metric("Temperature (last in range)", f"{m2:.2f} °C" if m2 is not None else "–", f"{d2:+.2f}" if d2 is not None else None)
    c3.metric("GDP growth (last in range)", f"{m3:.2f}%" if m3 is not None else "–", f"{d3:+.2f}" if d3 is not None else None)

    st.markdown("---")
    st.write("Use the tabs to explore each dataset. Upload your CSVs in the sidebar to replace the demo data.")

# -----------------------------
# Emissions Tab
# -----------------------------
with tab_emissions:
    st.subheader("Emissions over time")
    if "country" in em.columns:
        emc = em[em["country"].str.lower() == str(country).lower()].copy()
    else:
        emc = em.copy()
    if not emc.empty and "year" in emc.columns:
        emc = ensure_numeric(emc, ["year", "value"]).dropna(subset=["year", "value"])    
        emc = emc[(emc["year"] >= yr[0]) & (emc["year"] <= yr[1])]
        if not emc.empty:
            # Moving average
            emc = emc.sort_values("year")
            if smooth > 1:
                emc["value_smooth"] = emc["value"].rolling(window=smooth, min_periods=1, center=True).mean()
            else:
                emc["value_smooth"] = emc["value"]

            # Altair line chart
            y_enc = alt.Y("value_smooth:Q", title="Emissions")
            if log_scale:
                y_enc = alt.Y("value_smooth:Q", title="Emissions", scale=alt.Scale(type="log"))
            chart = (
                alt.Chart(emc)
                .mark_line(point=True)
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=y_enc,
                    tooltip=["year", alt.Tooltip("value_smooth", title="Emissions")],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)

            # Matplotlib example (from slides idea)
            fig, ax = plt.subplots()
            ax.plot(emc["year"], emc["value_smooth"], linewidth=2)
            ax.set_xlabel("Year")
            ax.set_ylabel("Emissions")
            ax.set_title(f"{country} Emissions (smoothed)")
            if log_scale:
                ax.set_yscale("log")
            st.pyplot(fig)
        else:
            st.info("No emissions data in selected range.")
    else:
        st.info("Emissions data missing required columns.")

    if show_raw:
        st.dataframe(em)

# -----------------------------
# Temperature Tab
# -----------------------------
with tab_temp:
    st.subheader("Annual mean temperature")
    if "country" in tp.columns:
        tpc = tp[tp["country"].str.lower() == str(country).lower()].copy()
    else:
        tpc = tp.copy()
    if not tpc.empty and "year" in tpc.columns and "value" in tpc.columns:
        tpc = ensure_numeric(tpc, ["year", "value"]).dropna(subset=["year", "value"])    
        tpc = tpc[(tpc["year"] >= yr[0]) & (tpc["year"] <= yr[1])]
        if not tpc.empty:
            tpc = tpc.sort_values("year")
            if smooth > 1:
                tpc["value_smooth"] = tpc["value"].rolling(window=smooth, min_periods=1, center=True).mean()
            else:
                tpc["value_smooth"] = tpc["value"]

            pxfig = px.line(tpc, x="year", y="value_smooth", markers=True, title=f"{country} Mean Temperature (°C)")
            pxfig.update_layout(yaxis_title="Temperature (°C)", xaxis_title="Year")
            st.plotly_chart(pxfig, use_container_width=True)
        else:
            st.info("No temperature data in selected range.")
    else:
        st.info("Temperature data missing required columns.")

    if show_raw:
        st.dataframe(tp)

# -----------------------------
# GDP Growth Tab
# -----------------------------
with tab_gdp:
    st.subheader("GDP growth over time")
    if "country" in gx.columns:
        gxc = gx[gx["country"].str.lower() == str(country).lower()].copy()
    else:
        gxc = gx.copy()
    if not gxc.empty and "year" in gxc.columns and "gdp_growth" in gxc.columns:
        gxc = ensure_numeric(gxc, ["year", "gdp_growth"]).dropna(subset=["year", "gdp_growth"])    
        gxc = gxc[(gxc["year"] >= yr[0]) & (gxc["year"] <= yr[1])]
        if not gxc.empty:
            gxc = gxc.sort_values("year")
            if smooth > 1:
                gxc["growth_smooth"] = gxc["gdp_growth"].rolling(window=smooth, min_periods=1, center=True).mean()
            else:
                gxc["growth_smooth"] = gxc["gdp_growth"]

            chart = (
                alt.Chart(gxc)
                .mark_line(point=True)
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("growth_smooth:Q", title="GDP growth (%)"),
                    tooltip=["year", alt.Tooltip("growth_smooth", title="GDP growth (%)")],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No GDP data in selected range.")
    else:
        st.info("GDP data missing required columns.")

    if show_raw:
        st.dataframe(gx)

# -----------------------------
# Disasters Tab
# -----------------------------
with tab_dis:
    st.subheader("Disaster counts per year")
    if not ds.empty and "year" in ds.columns:
        dsc = ds.copy()
        dsc = ensure_numeric(dsc, ["year"])    
        dsc = dsc[(dsc["year"] >= yr[0]) & (dsc["year"] <= yr[1])]
        if not dsc.empty:
            # Melt for stacked area / line per type
            value_cols = [c for c in dsc.columns if c != "year"]
            melted = dsc.melt(id_vars="year", value_vars=value_cols, var_name="type", value_name="count")
            melted = ensure_numeric(melted, ["count"]).dropna(subset=["count"])    

            chart = (
                alt.Chart(melted)
                .mark_area(opacity=0.6)
                .encode(
                    x=alt.X("year:O", title="Year"),
                    y=alt.Y("count:Q", stack="normalize", title="Share of total"),
                    color=alt.Color("type:N", title="Disaster type"),
                    tooltip=["year", "type", "count"],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)

            # Totals line
            totals = melted.groupby("year", as_index=False)["count"].sum()
            pxfig = px.bar(totals, x="year", y="count", title="Total disasters per year")
            st.plotly_chart(pxfig, use_container_width=True)
        else:
            st.info("No disasters data in selected range.")
    else:
        st.info("Disasters data missing required columns.")

    if show_raw:
        st.dataframe(ds)

# -----------------------------
# Footer / Help
# -----------------------------
with st.expander("ℹ️ Help & Data Schemas"):
    st.markdown(
        textwrap.dedent(
            """
            **CSV schemas**
            - *Emissions*: `Country, Year, Value` (or `Emissions` -> will be renamed to `Value`)
            - *Temperature*: `Country, Year, Value`
            - *GDP Growth*: `Country, Year, gdp_growth` (or `Value` -> will be renamed to `gdp_growth`)
            - *Disasters*: `Year, <one column per disaster type>`

            **Tips**
            - Use the sidebar to upload your CSVs and set filters.
            - Increase the moving average window to smooth noisy series.
            - Turn on *Log scale* for emissions with large ranges.
            """
        )
    )

