import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

YEAR_COLS = ["2022", "2023", "2024", "2025"]
MONEY_COLS = YEAR_COLS + ["Grand Total"]

# ---------------- HELPERS ----------------

def parse_money_val(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip().replace("$","").replace(",","")
    if s in ["","-","nan","None"]:
        return np.nan
    if "(" in s and ")" in s:
        s = "-" + s.replace("(","").replace(")","")
    try:
        return float(s)
    except:
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
            yoy_pct = (yoy_change / prev) * 100
        data.append({
            "Year": int(year),
            "Sales": val,
            "YoY $ Change": yoy_change,
            "YoY % Change": yoy_pct,
        })
        prev = val
    return pd.DataFrame(data)

def format_money(x):
    return "" if pd.isna(x) else f"${x:,.2f}"

def format_pct(x):
    return "" if pd.isna(x) else f"{x:+.1f}%"

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

    y = height - 75
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Customer Performance Report")
    y -= 30

    c.setFont("Helvetica", 11)

    # --- Customer details ---
    c.drawString(50, y, f"Customer: {row['Customer Name']}")
    y -= 18
    c.drawString(50, y, f"Customer #: {row['Cust. #']}")
    y -= 18
    c.drawString(50, y, f"Sales Rep: {row['Outside Rep']}")
    y -= 18
    c.drawString(50, y, f"City: {row.get('City', '')}  State: {row.get('State', '')}")
    y -= 18
    c.drawString(50, y, f"Industry: {row.get('Industry', '')}")
    y -= 25

    # --- Yearly Table ---
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Yearly Sales Summary")
    y -= 22
    c.setFont("Helvetica", 11)

    for _, r in df_years.iterrows():
        c.drawString(
            50,
            y,
            f"{int(r['Year'])}: {format_money(r['Sales'])}"
        )
        y -= 18

    # --- Chart ---
    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    y = 200
    c.drawImage(ImageReader(chart_buf), 50, y, width=500, preserveAspectRatio=True)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- STREAMLIT APP ----------------

st.set_page_config(page_title="Customer Sales Analytics", layout="wide")
st.title("📈 Customer Sales Analytics")

CSV_URL = "https://raw.githubusercontent.com/SystemAdmin-RP/customer-sales-analytics/main/CustomerTrend.csv"

admin_pass = st.sidebar.text_input("Admin Password", type="password")

# Admin upload (optional override)
if admin_pass == "admin123":
    uploaded = st.sidebar.file_uploader("Upload CustomerTrend CSV", type="csv")
else:
    uploaded = None

# If admin uploaded: use that. Otherwise: load GitHub CSV.
if uploaded:
    df = pd.read_csv(uploaded, encoding="latin1")
else:
    df = pd.read_csv(CSV_URL, encoding="latin1")

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
    results["Customer Name"] + " (#" + results["Cust. #"].astype(str) + ")"
)

row = results.iloc[
    (results["Customer Name"] + " (#" + results["Cust. #"].astype(str) + ")"
     ).tolist().index(selected)
]

df_years = build_yearly_table(row)
fig = create_trend_figure(df_years, row["Customer Name"])
st.pyplot(fig)

st.dataframe(df_years.style.format({
    "Sales": format_money,
    "YoY $ Change": format_money,
    "YoY % Change": format_pct,
}), hide_index=True)

pdf = generate_pdf_report(row, df_years, fig)

st.download_button(
    "Download PDF Report",
    data=pdf,
    file_name=f"{row['Cust. #']}_report.pdf",
    mime="application/pdf",
)
