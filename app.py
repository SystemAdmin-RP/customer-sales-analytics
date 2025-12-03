import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from reportlab.lib.pagesizes import letter, landscape
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
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_years["Year"], df_years["Sales"], marker="o")
    ax.set_title(f"Year-over-Year Sales Trend - {customer_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.grid(True)

    # Clean ticks
    ax.set_xticks(df_years["Year"])
    ax.get_yaxis().set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    fig.tight_layout()
    return fig

# ---------------- PDF: SINGLE CUSTOMER ----------------

def generate_pdf_report(row, df_years, fig):
    buffer = io.BytesIO()

    # Landscape
    page = landscape(letter)
    width, height = page
    c = canvas.Canvas(buffer, pagesize=page)

    # HEADER
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width / 2, height - 45, "Regal Plastics")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 75, "Customer Performance Report")

    left_x = 40
    top_y = height - 120

    # CUSTOMER INFO
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, top_y, "Customer Information")

    y = top_y - 22
    c.setFont("Helvetica", 11)
    c.drawString(left_x, y, f"Customer:   {row['Customer Name']}")
    y -= 16
    c.drawString(left_x, y, f"Customer #: {row['Cust. #']}")
    y -= 16
    c.drawString(left_x, y, f"Sales Rep:  {row.get('Outside Rep','')}")
    y -= 16
    c.drawString(left_x, y, f"City:       {row.get('City','')}   State: {row.get('State','')}")
    y -= 16
    c.drawString(left_x, y, f"Industry:   {row.get('Industry','')}")

    # KPI VALUES
    total_sales = df_years["Sales"].sum()
    best = df_years.loc[df_years["Sales"].idxmax()]
    worst = df_years.loc[df_years["Sales"].idxmin()]
    df2 = df_years.dropna(subset=["YoY % Change"])
    avg_growth = df2["YoY % Change"].mean() if len(df2) > 0 else None

    kpi_data = [
        ("Total Sales", format_money(total_sales)),
        ("Best Year", f"{int(best['Year'])}  ({format_money(best['Sales'])})"),
        ("Worst Year", f"{int(worst['Year'])} ({format_money(worst['Sales'])})"),
        ("Avg Growth", f"{avg_growth:.1f}%" if avg_growth is not None else "N/A"),
    ]

    # KPI CARDS STACKED (FULL WIDTH)
    card_x = left_x
    card_w = width - 2 * left_x
    card_h = 42
    spacing = 8
    kpi_y = y - 24

    for title, value in kpi_data:
        c.setFillColorRGB(0.95, 0.95, 0.95)
        c.rect(card_x, kpi_y - card_h, card_w, card_h, fill=1, stroke=0)

        c.setStrokeColorRGB(0.75, 0.75, 0.75)
        c.rect(card_x, kpi_y - card_h, card_w, card_h, fill=0, stroke=1)

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(card_x + 10, kpi_y - 14, title)

        c.setFont("Helvetica", 11)
        c.drawString(card_x + 10, kpi_y - 30, value)

        kpi_y -= (card_h + spacing)

    # YEARLY TABLE
    table_y = kpi_y - 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, table_y, "Yearly Performance")
    table_y -= 18

    headers = ["Year", "Sales", "YoY $", "YoY %"]
    col_x = [left_x, left_x + 70, left_x + 190, left_x + 290]

    c.setFillColorRGB(0.88, 0.88, 0.88)
    c.rect(left_x, table_y - 3, 360, 16, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 10)
    for i, h in enumerate(headers):
        c.drawString(col_x[i], table_y, h)

    table_y -= 16

    c.setFont("Helvetica", 10)
    for i, (_, r) in enumerate(df_years.iterrows()):
        # stop table early if too low
        if table_y < 220:
            break

        if i % 2 == 0:
            c.setFillColorRGB(0.96, 0.96, 0.96)
            c.rect(left_x, table_y - 2, 360, 14, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)

        c.drawString(col_x[0], table_y, str(int(r["Year"])))
        c.drawString(col_x[1], table_y, format_money(r["Sales"]))
        c.drawString(col_x[2], table_y,
                     format_money(r["YoY $ Change"]) if not pd.isna(r["YoY $ Change"]) else "")
        c.drawString(col_x[3], table_y,
                     format_pct(r["YoY % Change"]) if not pd.isna(r["YoY % Change"]) else "")

        table_y -= 14

    # CHART FORMATTING
    ax = fig.axes[0]
    ax.set_xticks(df_years["Year"])
    ax.get_yaxis().set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    # DRAW CHART (BOTTOM, SMALLER HEIGHT)
    c.drawImage(
        ImageReader(chart_buf),
        left_x,
        40,
        width=width - 2 * left_x,
        height=180,
        preserveAspectRatio=True
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ---------------- PDF: MULTI-COMPARISON ----------------

def generate_multi_pdf(selected_rows, fig, comparison_df_num):
    buffer = io.BytesIO()

    page = landscape(letter)
    width, height = page
    c = canvas.Canvas(buffer, pagesize=page)

    # HEADER
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width / 2, height - 45, "Regal Plastics")
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 75, "Customer Comparison Report")

    left_x = 40
    y = height - 115

    # SUMMARY OVERVIEW
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, y, "Summary Overview")
    y -= 22

    c.setFont("Helvetica", 11)
    for r in selected_rows:
        total = comparison_df_num.loc[r["Customer Name"], "Total"]
        line = f"{r['Customer Name']}  |  #{r['Cust. #']}  |  Total Sales: {format_money(total)}"
        c.drawString(left_x, y, line)
        y -= 16

    # KPI CARDS STACKED FULL WIDTH
    totals_sorted = comparison_df_num["Total"].sort_values(ascending=False)
    best_name = totals_sorted.index[0]
    best_val = totals_sorted.iloc[0]
    worst_name = totals_sorted.index[-1]
    worst_val = totals_sorted.iloc[-1]
    combined_total = comparison_df_num["Total"].sum()
    avg_per_cust = combined_total / len(comparison_df_num) if len(comparison_df_num) > 0 else 0

    kpi_data = [
        ("Top Customer", f"{best_name} ({format_money(best_val)})"),
        ("Lowest Customer", f"{worst_name} ({format_money(worst_val)})"),
        ("Total of Selected", format_money(combined_total)),
        ("Avg per Customer", format_money(avg_per_cust)),
    ]

    card_x = left_x
    card_w = width - 2 * left_x
    card_h = 42
    spacing = 8
    kpi_y = y - 10

    for title, value in kpi_data:
        c.setFillColorRGB(0.95, 0.95, 0.95)
        c.rect(card_x, kpi_y - card_h, card_w, card_h, fill=1, stroke=0)

        c.setStrokeColorRGB(0.75, 0.75, 0.75)
        c.rect(card_x, kpi_y - card_h, card_w, card_h, fill=0, stroke=1)

        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(card_x + 10, kpi_y - 14, title)

        c.setFont("Helvetica", 11)
        c.drawString(card_x + 10, kpi_y - 30, value)

        kpi_y -= (card_h + spacing)

    # TABLE HEADER
    table_y = kpi_y - 24
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, table_y, "Sales Comparison Table")
    table_y -= 18

    columns = comparison_df_num.columns.tolist()  # e.g. [2022, 2023, ..., Total]

    name_col_width = 230
    other_col_width = 80

    col_x_positions = [left_x]
    for i in range(len(columns)):
        if i == 0:
            col_x_positions.append(col_x_positions[-1] + name_col_width)
        else:
            col_x_positions.append(col_x_positions[-1] + other_col_width)

    header_height = 18
    c.setFillColorRGB(0.88, 0.88, 0.88)
    c.rect(left_x, table_y - 4,
           col_x_positions[-1] - left_x + other_col_width,
           header_height, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(col_x_positions[0], table_y, "Customer")
    for i, col in enumerate(columns):
        c.drawString(col_x_positions[i + 1], table_y, str(col))

    table_y -= header_height

    # TABLE ROWS
    row_h = 16
    c.setFont("Helvetica", 9)

    for idx, cust in enumerate(comparison_df_num.index):
        if table_y < 230:  # leave room for chart
            break

        if idx % 2 == 0:
            c.setFillColorRGB(0.96, 0.96, 0.96)
            c.rect(left_x, table_y - 2,
                   col_x_positions[-1] - left_x + other_col_width,
                   row_h, fill=1, stroke=0)
            c.setFillColorRGB(0, 0, 0)

        display_name = cust if len(cust) <= 30 else cust[:28] + "…"
        c.drawString(col_x_positions[0], table_y, display_name)

        for i, col in enumerate(columns):
            val = comparison_df_num.loc[cust, col]
            s = format_money(val) if not pd.isna(val) else ""
            c.drawRightString(col_x_positions[i + 1] + other_col_width - 6, table_y, s)

        table_y -= row_h

    # CHART IMAGE
    chart_buf = io.BytesIO()
    fig.savefig(chart_buf, format="png", bbox_inches="tight")
    chart_buf.seek(0)

    c.drawImage(
        ImageReader(chart_buf),
        left_x,
        40,
        width=width - 2 * left_x,
        height=180,
        preserveAspectRatio=True
    )

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
        selected_rows = []
        for choice in selected_customers:
            r = df.iloc[(all_options == choice).to_numpy().nonzero()[0][0]]
            selected_rows.append(r)

        # comparison chart
        fig2, ax = plt.subplots(figsize=(6, 3))

        comparison_rows = []
        for r in selected_rows:
            dfy = build_yearly_table(r)
            ax.plot(dfy["Year"], dfy["Sales"], marker="o", label=r["Customer Name"])

            row_data = {int(y): v for y, v in zip(dfy["Year"], dfy["Sales"])}
            row_data["Customer"] = r["Customer Name"]
            row_data["Total"] = dfy["Sales"].sum()
            comparison_rows.append(row_data)

        all_years = sorted({int(y) for y in YEAR_COLS})
        ax.set_xticks(all_years)
        ax.get_yaxis().set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

        ax.set_title("Customer Comparison Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sales")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig2)

        comparison_df_num = pd.DataFrame(comparison_rows).set_index("Customer")
        comparison_df_display = comparison_df_num.applymap(
            lambda x: format_money(x) if not pd.isna(x) else ""
        )

        st.subheader("Sales Comparison Table")
        st.dataframe(comparison_df_display)

        st.subheader("Comparison KPIs")
        totals_sorted = comparison_df_num["Total"].sort_values(ascending=False)

        st.write(
            f"🏆 Best Customer: {totals_sorted.index[0]} — {format_money(totals_sorted.iloc[0])}"
        )
        st.write(
            f"📉 Lowest Customer: {totals_sorted.index[-1]} — {format_money(totals_sorted.iloc[-1])}"
        )

        pdf_multi = generate_multi_pdf(selected_rows, fig2, comparison_df_num)
        st.download_button(
            "Download Comparison PDF",
            data=pdf_multi,
            file_name="comparison_report.pdf",
            mime="application/pdf",
        )
