import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
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

# ---------------- PDF FUNCTIONS ----------------

def generate_pdf_report(row, df_years, fig):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Regal Plastics")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-90, "Customer Performance Report")

    y = height - 120
    c.setFont("Helvetica", 11)

    c.drawString(50, y, f"Customer   : {row['Customer Name']}")
    y -= 16
    c.drawString(50, y, f"Customer # : {row['Cust. #']}")
    y -= 16
    c.drawString(50, y, f"Sales Rep  : {row.get('Outside Rep','')}")
    y -= 16
    c.drawString(50, y, f"City       : {row.get('City','')}  State: {row.get('State','')}")
    y -= 16
    c.drawString(50, y, f"Industry   : {row.get('Industry','')}")
    y -= 24

    total_sales = df_years["Sales"].sum()
    best = df_years.loc[df_years["Sales"].idxmax()]
    worst = df_years.loc[df_years["Sales"].idxmin()]

    df2 = df_years.dropna(subset=["YoY % Change"])
    avg_growth = df2["YoY % Change"].mean() if len(df2)>0 else None

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Summary KPIs")
    y -= 20
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Total Sales: {format_money(total_sales)}")
    y -= 16
    c.drawString(50, y, f"Best Year: {int(best['Year'])}  ({format_money(best['Sales'])})")
    y -= 16
    c.drawString(50, y, f"Worst Year: {int(worst['Year'])} ({format_money(worst['Sales'])})")
    y -= 16
    if avg_growth:
        c.drawString(50, y, f"Average Growth: {avg_growth:.1f}%")
        y -= 24

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Year   Sales              YoY$       YoY%")
    y -= 18
    c.line(50, y, width-50, y)
    y -= 12

    c.setFont("Helvetica", 11)
    for _, r in df_years.iterrows():
        line = f"{int(r['Year'])}   {format_money(r['Sales'])}"
        if not pd.isna(r['YoY $ Change']):
            line += f"     {format_money(r['YoY $ Change'])}     {format_pct(r['YoY % Change'])}"
        c.drawString(50, y, line)
        y -= 16

    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    c.drawImage(
        ImageReader(chart_buf),
        50, 80,
        width=500, height=250,
        preserveAspectRatio=True
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


def generate_multi_pdf(selected_rows, fig):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height-50, "Regal Plastics – Comparison Report")

    y = height - 90
    c.setFont("Helvetica", 10)

    for r in selected_rows:
        c.drawString(50, y, f"{r['Customer Name']} | #{r['Cust. #']} | {format_money(r['Grand Total'])}")
        y -= 14

    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    c.drawImage(ImageReader(chart_buf), 50, 100, width=500, height=250)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- APP START ----------------

st.set_page_config(page_title="Customer Sales Analytics", layout="wide")
st.title("📈 Customer Sales Analytics")

CSV_URL = "https://raw.githubusercontent.com/SystemAdmin-RP/customer-sales-analytics/main/CustomerTrend.csv"

admin_pass = st.sidebar.text_input("Admin Password", type="password")
uploaded = None

if admin_pass == "admin123":
    uploaded = st.sidebar.file_uploader("Upload CustomerTrend CSV", type="csv")

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

tab1, tab2 = st.tabs(["Single Customer View", "Comparison View"])

# ---------------- TAB 1: SINGLE CUSTOMER ----------------

with tab1:

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

# ---------------- TAB 2: MULTI-COMPARISON ----------------

with tab2:

    st.subheader("Compare multiple customers")

    selected_customers = st.multiselect(
        "Select 2–10 customers:",
        results["Customer Name"] + " (#" + results["Cust. #"].astype(str) + ")"
    )

    if len(selected_customers) < 2:
        st.stop()

    selected_rows = []
    for choice in selected_customers:
        r = results.iloc[
            (results["Customer Name"] + " (#" + results["Cust. #"].astype(str) + ")"
             ).tolist().index(choice)
        ]
        selected_rows.append(r)

    fig2, ax = plt.subplots(figsize=(8,4))

    for r in selected_rows:
        dfy = build_yearly_table(r)
        ax.plot(dfy["Year"], dfy["Sales"], marker="o", label=r["Customer Name"])

    ax.set_title("Customer Comparison Trend")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig2)

    comparison_df = pd.DataFrame()

    for r in selected_rows:
        dfy = build_yearly_table(r)
        row_data = {int(y): v for y,v in zip(dfy["Year"], dfy["Sales"])}
        row_data["Customer"] = r["Customer Name"]
        row_data["Total"] = dfy["Sales"].sum()
        comparison_df = comparison_df.append(row_data, ignore_index=True)

    comparison_df = comparison_df.set_index("Customer")
    comparison_df = comparison_df.applymap(lambda x: format_money(x) if not pd.isna(x) else "")
    st.subheader("Sales Comparison Table")
    st.dataframe(comparison_df)

    # KPIs
    st.subheader("Comparison KPIs")
    totals_sorted = comparison_df["Total"].str.replace("$","").str.replace(",","").astype(float).sort_values(ascending=False)

    st.write(f"🏆 Best Customer: {totals_sorted.index[0]} — {format_money(totals_sorted.iloc[0])}")
    st.write(f"📉 Lowest Customer: {totals_sorted.index[-1]} — {format_money(totals_sorted.iloc[-1])}")

    pdf_multi = generate_multi_pdf(selected_rows, fig2)
    st.download_button(
        "Download Comparison PDF",
        data=pdf_multi,
        file_name="comparison_report.pdf",
        mime="application/pdf",
    )
