import io
import re

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


# ---------- Helpers ----------

YEAR_COLS = ["2022", "2023", "2024", "2025"]
MONEY_COLS = YEAR_COLS + ["Grand Total"]


def parse_money_val(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s in ["", "-", "nan", "None", "$-", "$-   "]:
        return np.nan
    s = s.replace("$", "").replace(",", "").strip()
    m = re.fullmatch(r"\((.*)\)", s)
    if m:
        s = "-" + m.group(1)
    try:
        return float(s)
    except Exception:
        return np.nan


def build_yearly_table(row):
    data = []
    prev = None
    for year in YEAR_COLS:
        val = row.get(year, np.nan)
        if pd.isna(val):
            continue
        yoy_change = None
        yoy_pct = None
        if prev is not None and prev != 0:
            yoy_change = val - prev
            yoy_pct = (yoy_change / prev) * 100.0
        data.append(
            {
                "Year": int(year),
                "Sales": val,
                "YoY $ Change": yoy_change,
                "YoY % Change": yoy_pct,
            }
        )
        prev = val

    if not data:
        return pd.DataFrame(columns=["Year", "Sales", "YoY $ Change", "YoY % Change"])

    return pd.DataFrame(data)


def format_money(x):
    if pd.isna(x):
        return ""
    return f"${x:,.2f}"


def format_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:+.1f}%"


def create_trend_figure(df_years, customer_name):
    fig, ax = plt.subplots()
    ax.plot(df_years["Year"], df_years["Sales"], marker="o")
    ax.set_title(f"Year-over-Year Sales Trend - {customer_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.grid(True)
    fig.tight_layout()
    return fig


def generate_pdf_report(row, df_years, fig):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    name = str(row.get("Customer Name", "")).strip()
    num = str(row.get("Cust. #", "")).strip()
    rep = str(row.get("Outside Rep", "")).strip()
    grand_total = row.get("Grand Total", np.nan)

    y = height - 75
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Customer Performance Report")
    y -= 35

    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Customer: {name}")
    y -= 18
    c.drawString(50, y, f"Customer #: {num}")
    y -= 18
    c.drawString(50, y, f"Rep: {rep}")
    y -= 18
    c.drawString(50, y, f"Grand Total: {format_money(grand_total)}")
    y -= 35

    for _, r in df_years.iterrows():
        c.drawString(50, y, f"{int(r['Year'])}: {format_money(r['Sales'])}")
        y -= 18

    buffer.seek(0)
    c.save()
    return buffer.getvalue()


# ---------- Streamlit App ----------

st.set_page_config(page_title="Customer Sales Analytics", layout="wide")

st.title("📈 Customer Sales Analytics")

uploaded = st.file_uploader("Upload CustomerTrend CSV", type="csv")

if uploaded is None:
    st.info("Upload your CSV above to begin.")
    st.stop()

df = pd.read_csv(uploaded, encoding="latin1")

for col in MONEY_COLS:
    if col in df.columns:
        df[col] = df[col].map(parse_money_val)

query = st.text_input("Search customer name or number:")

if not query:
    st.stop()

results = df[
    df["Customer Name"].astype(str).str.contains(query, case=False)
    | df["Cust. #"].astype(str).str.contains(query, case=False)
]

if results.empty:
    st.warning("No match found.")
    st.stop()

selected = st.selectbox(
    "Select a customer:",
    results["Customer Name"] + " (#" + results["Cust. #"].astype(str) + ")",
)

row = results.iloc[[i for i, v in enumerate(
    results["Customer Name"] + " (#" + results["Cust. #"].astype(str) + ")"
) if v == selected][0]]

df_years = build_yearly_table(row)
fig = create_trend_figure(df_years, row["Customer Name"])

st.pyplot(fig)

st.write(df_years.style.format({
    "Sales": format_money,
    "YoY $ Change": format_money,
    "YoY % Change": format_pct,
}))

pdf = generate_pdf_report(row, df_years, fig)

st.download_button(
    "Download PDF Report",
    data=pdf,
    file_name=f"Report_{row['Cust. #']}.pdf",
    mime="application/pdf",
)
