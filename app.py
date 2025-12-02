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
    s = str(x).strip().replace("$", "").replace(",", "")
    if s in ["", "-", "nan", "None"]:
        return np.nan
    if "(" in s and ")" in s:
        s = "-" + s.replace("(", "").replace(")", "")
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

    # Title
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width / 2, height - 50, "Regal Plastics")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 90, "Customer Performance Report")

    y = height - 120
    c.setFont("Helvetica", 11)

    # Customer details
    c.drawString(50, y, f"Customer   : {row['Customer Name']}")
    y -= 16
    c.drawString(50, y, f"Customer # : {row['Cust. #']}")
    y -= 16
    c.drawString(50, y, f"Sales Rep  : {row.get('Outside Rep', '')}")
    y -= 16
    c.drawString(50, y, f"City       : {row.get('City', '')}  State: {row.get('State', '')}")
    y -= 16
    c.drawString(50, y, f"Industry   : {row.get('Industry', '')}")
    y -= 24

    # KPIs
    total_sales = df_years["Sales"].sum()
    best = df_years.loc[df_years["Sales"].idxmax()]
    worst = df_years.loc[df_years["Sales"].idxmin()]

    df2 = df_years.dropna(subset=["YoY % Change"])
    avg_growth = df2["YoY % Change"].mean() if len(df2) > 0 else None

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
    if avg_growth is not None:
        c.drawString(50, y, f"Average Growth: {avg_growth:.1f}%")
        y -= 24

    # Yearly table header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Year   Sales              YoY$       YoY%")
    y -= 18
    c.line(50, y, width - 50, y)
    y -= 12

    # Table rows
    c.setFont("Helvetica", 11)
    for _, r in df_years.iterrows():
        line = f"{int(r['Year'])}   {format_money(r['Sales'])}"
        if not pd.isna(r['YoY $ Change']):
            line += f"     {format_money(r['YoY $ Change'])}     {format_pct(r['YoY % Change'])}"
        c.drawString(50, y, line)
        y -= 16

    # Chart
    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    c.drawImage(
        ImageReader(chart_buf),
        50, 100,
        width=500, height=300,
        preserveAspectRatio=True
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

from reportlab.lib.pagesizes import landscape, letter

def generate_multi_pdf(selected_rows, fig, comparison_df_num):
    buffer = io.BytesIO()

    # Landscape page
    page = landscape(letter)
    width, height = page

    c = canvas.Canvas(buffer, pagesize=page)

    # HEADER
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, height - 50, "Regal Plastics")

    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(width / 2, height - 80, "Customer Comparison Report")

    y = height - 120

    # ======================
    # Summary Overview
    # ======================
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Summary Overview")
    y -= 25

    c.setFont("Helvetica", 11)
    for r in selected_rows:
        total = comparison_df_num.loc[r["Customer Name"], "Total"]
        line = f"{r['Customer Name']}  |  #{r['Cust. #']}  |  Total Sales: {format_money(total)}"
        c.drawString(40, y, line)
        y -= 16

    # Highest growth
    growth_by_cust = {}
    for r in selected_rows:
        dfy = build_yearly_table(r)
        dfy = dfy.dropna(subset=["YoY % Change"])
        growth_by_cust[r["Customer Name"]] = (
            dfy["YoY % Change"].mean() if len(dfy) else None
        )

    valid_growth = {k: v for k, v in growth_by_cust.items() if v is not None}
    y -= 10
    c.setFont("Helvetica-Bold", 11)

    if valid_growth:
        best_name = max(valid_growth, key=valid_growth.get)
        best_val = valid_growth[best_name]
        c.drawString(40, y, f"Highest Avg Growth: {best_name} ({best_val:.1f}%)")
    else:
        c.drawString(40, y, "Highest Avg Growth: N/A")
    y -= 28

    # ======================
    # Table Header
    # ======================
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Sales Comparison Table")
    y -= 18

    columns = comparison_df_num.columns.tolist()

    name_col_width = 230
    other_col_width = 95

    col_x_positions = [40]
    for i in range(len(columns)):
        if i == 0:
            col_x_positions.append(col_x_positions[-1] + name_col_width)
        else:
            col_x_positions.append(col_x_positions[-1] + other_col_width)

    # Header bg
    header_height = 18
    c.setFillColorRGB(0.88, 0.88, 0.88)
    c.rect(40, y - 4, col_x_positions[-1] - 40 + other_col_width, header_height, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(col_x_positions[0], y, "Customer")
    for i, col in enumerate(columns):
        c.drawString(col_x_positions[i + 1], y, str(col))

    y -= header_height

    # ======================
    # Table Rows
    # ======================
    row_h = 16
    c.setFont("Helvetica", 9)

    for idx, cust in enumerate(comparison_df_num.index):
        # Row shading for alternating lines
        if idx % 2 == 0:
            c.setFillColorRGB(0.96, 0.96, 0.96)
            c.rect(40, y - 2, col_x_positions[-1] - 40 + other_col_width, row_h, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)

        display_name = cust if len(cust) <= 30 else cust[:28] + "…"
        c.drawString(col_x_positions[0], y, display_name)

        for i, col in enumerate(columns):
            val = comparison_df_num.loc[cust, col]
            s = format_money(val) if not pd.isna(val) else ""
            c.drawRightString(col_x_positions[i + 1] + other_col_width - 6, y, s)

        y -= row_h

        # stop before chart area
        if y < 200:
            break

    y -= 30

    # ======================
    # CHART (bigger landscape)
    # ======================
    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    c.drawImage(
        ImageReader(chart_buf),
        60, 60,
        width=650,
        height=260,
        preserveAspectRatio=True
    )

    # END PAGE
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

query = st.text_input("Search customer name or number for SINGLE view:")

tab1, tab2 = st.tabs(["Single Customer View", "Comparison View"])

# ---------------- TAB 1: SINGLE CUSTOMER ----------------

with tab1:
    if not query:
        st.info("Search for a customer above.")
    else:
        results = df[
            df["Customer Name"].astype(str).str.contains(query, case=False)
            | df["Cust. #"].astype(str).str.contains(query, case=False)
        ]

        if results.empty:
            st.warning("No match found.")
        else:
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

            st.dataframe(
                df_years.style.format({
                    "Sales": format_money,
                    "YoY $ Change": format_money,
                    "YoY % Change": format_pct,
                }),
                hide_index=True
            )

            pdf = generate_pdf_report(row, df_years, fig)
            st.download_button(
                "Download PDF Report",
                data=pdf,
                file_name=f"{row['Cust. #']}_report.pdf",
                mime="application/pdf",
            )

# ---------------- TAB 2: MULTI-COMPARISON ----------------

with tab2:
    st.subheader("Compare multiple customers (any from the entire dataset)")

    all_options = df["Customer Name"].astype(str) + " (#" + df["Cust. #"].astype(str) + ")"

    selected_customers = st.multiselect(
        "Select 2–10 customers:",
        all_options
    )

    if len(selected_customers) == 0:
        st.info("Select at least 2 customers to see comparison.")
    elif len(selected_customers) < 2:
        st.warning("Please select at least 2 customers.")
    elif len(selected_customers) > 10:
        st.warning("Please select 10 or fewer customers.")
    else:
        # Build selected rows
        selected_rows = []
        for choice in selected_customers:
            r = df.iloc[(all_options == choice).to_numpy().nonzero()[0][0]]
            selected_rows.append(r)

        # Trend comparison chart
        fig2, ax = plt.subplots(figsize=(6, 3))

        comparison_rows = []
        for r in selected_rows:
            dfy = build_yearly_table(r)
            ax.plot(dfy["Year"], dfy["Sales"], marker="o", label=r["Customer Name"])

            row_data = {int(y): v for y, v in zip(dfy["Year"], dfy["Sales"])}
            row_data["Customer"] = r["Customer Name"]
            row_data["Total"] = dfy["Sales"].sum()
            comparison_rows.append(row_data)

        ax.set_title("Customer Comparison Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sales")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig2)

        # Comparison numeric + display tables
        comparison_df_num = pd.DataFrame(comparison_rows).set_index("Customer")
        comparison_df_display = comparison_df_num.applymap(
            lambda x: format_money(x) if not pd.isna(x) else ""
        )

        st.subheader("Sales Comparison Table")
        st.dataframe(comparison_df_display)

        # KPIs
        st.subheader("Comparison KPIs")
        totals_sorted = comparison_df_num["Total"].sort_values(ascending=False)

        st.write(
            f"🏆 Best Customer: {totals_sorted.index[0]} — {format_money(totals_sorted.iloc[0])}"
        )
        st.write(
            f"📉 Lowest Customer: {totals_sorted.index[-1]} — {format_money(totals_sorted.iloc[-1])}"
        )

        # Comparison PDF
        pdf_multi = generate_multi_pdf(selected_rows, fig2, comparison_df_num)
        st.download_button(
            "Download Comparison PDF",
            data=pdf_multi,
            file_name="comparison_report.pdf",
            mime="application/pdf",
        )
