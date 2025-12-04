import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

# ---------------- CONSTANTS ----------------

YEAR_COLS = ["2022", "2023", "2024", "2025"]
MONEY_COLS = YEAR_COLS + ["Grand Total"]

REGAL_BLUE = colors.HexColor("#003B70")
REGAL_LIGHT_BLUE = colors.HexColor("#E8F1FA")


# ---------------- SAFE CSV LOADING ----------------

def safe_read_csv(source):
    """
    Attempts multiple encodings to safely load any CSV
    exported from Excel, ERP systems, Power Query, or GitHub.
    """
    encodings = ["utf-8-sig", "latin1", "ISO-8859-1"]

    for enc in encodings:
        try:
            return pd.read_csv(source, encoding=enc, engine="python")
        except Exception:
            continue

    # Absolute fallback
    return pd.read_csv(source, engine="python", errors="ignore")


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
            "YoY % Change": yoy_pct
        })

        prev = val

    return pd.DataFrame(data)


def format_money(x):
    return "" if pd.isna(x) else f"${x:,.2f}"


def format_pct(x):
    return "" if pd.isna(x) else f"{x:+.1f}%"


def create_trend_figure(df_years, name):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_years["Year"], df_years["Sales"], marker="o", label=name)

    ax.set_title(f"Year-over-Year Sales Trend - {name}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    ax.grid(True)

    # Clean ticks
    ax.set_xticks(df_years["Year"])
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------- PDF: SINGLE CUSTOMER ----------------

def generate_pdf_report(row, df_years, fig):
    buffer = io.BytesIO()

    page = landscape(letter)
    width, height = page
    c = canvas.Canvas(buffer, pagesize=page)

    # HEADER
    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(REGAL_BLUE)
    c.drawCentredString(width / 2, height - 45, "Regal Plastics")

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 75, "Customer Performance Report")

    c.setFillColor(colors.black)

    left_x = 40
    right_margin = 40
    content_width = width - left_x - right_margin
    top_y = height - 115

    # CUSTOMER INFO (full width)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, top_y, "Customer Information")
    y = top_y - 22

    c.setFont("Helvetica", 11)
    c.drawString(left_x, y, f"Customer:   {row['Customer Name']}"); y -= 16
    c.drawString(left_x, y, f"Customer #: {row['Cust. #']}"); y -= 16
    c.drawString(left_x, y, f"Sales Rep:  {row.get('Outside Rep','')}"); y -= 16
    c.drawString(left_x, y, f"City:       {row.get('City','')}   State: {row.get('State','')}"); y -= 16
    c.drawString(left_x, y, f"Industry:   {row.get('Industry','')}")

    # KPI VALUES
    total_sales = df_years["Sales"].sum()
    best = df_years.loc[df_years["Sales"].idxmax()]
    worst = df_years.loc[df_years["Sales"].idxmin()]
    avg_growth = df_years["YoY % Change"].dropna().mean()
    if pd.isna(avg_growth):
        avg_growth_str = "N/A"
    else:
        avg_growth_str = f"{avg_growth:.1f}%"

    kpis = [
        ("Total Sales", format_money(total_sales)),
        ("Best Year", f"{int(best['Year'])} ({format_money(best['Sales'])})"),
        ("Worst Year", f"{int(worst['Year'])} ({format_money(worst['Sales'])})"),
        ("Avg Growth", avg_growth_str),
    ]

    # MID SECTION: 2 COLUMNS (50/50)
    mid_top_y = y - 30
    col_gap = 20
    col_width = (content_width - col_gap) / 2

    kpi_col_x = left_x
    table_col_x = left_x + col_width + col_gap

    # KPI CARDS in LEFT COLUMN (smaller, blue theme)
    card_w = col_width
    card_h = 26
    spacing = 4
    kpi_y = mid_top_y

    for title, value in kpis:
        # background
        c.setFillColor(REGAL_LIGHT_BLUE)
        c.rect(kpi_col_x, kpi_y - card_h, card_w, card_h, fill=1, stroke=0)
        # border
        c.setStrokeColor(REGAL_BLUE)
        c.rect(kpi_col_x, kpi_y - card_h, card_w, card_h, fill=0, stroke=1)

        # text
        c.setFillColor(REGAL_BLUE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(kpi_col_x + 8, kpi_y - 10, title)

        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(kpi_col_x + 8, kpi_y - 21, value)

        kpi_y -= (card_h + spacing)

    # YEARLY PERFORMANCE TABLE in RIGHT COLUMN
    table_top_y = mid_top_y
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(REGAL_BLUE)
    c.drawString(table_col_x, table_top_y, "Yearly Performance")
    c.setFillColor(colors.black)

    table_y = table_top_y - 18

    headers = ["Year", "Sales", "YoY $", "YoY %"]
    col_x = [
        table_col_x,
        table_col_x + 60,
        table_col_x + 160,
        table_col_x + 260,
    ]

    c.setFillColorRGB(0.88, 0.88, 0.88)
    c.rect(table_col_x, table_y - 3, 260, 16, fill=1, stroke=0)
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 9)
    for i, h in enumerate(headers):
        c.drawString(col_x[i], table_y, h)

    table_y -= 16

    c.setFont("Helvetica", 9)
    for i, (_, r) in enumerate(df_years.iterrows()):
        if i % 2 == 0:
            c.setFillColorRGB(0.96, 0.96, 0.96)
            c.rect(table_col_x, table_y - 2, 260, 14, fill=1, stroke=0)
            c.setFillColor(colors.black)

        c.drawString(col_x[0], table_y, str(int(r["Year"])))
        c.drawString(col_x[1], table_y, format_money(r["Sales"]))
        c.drawString(col_x[2], table_y,
                     format_money(r["YoY $ Change"]) if not pd.isna(r["YoY $ Change"]) else "")
        c.drawString(col_x[3], table_y,
                     format_pct(r["YoY % Change"]) if not pd.isna(r["YoY % Change"]) else "")

        table_y -= 14

    # CHART (full width at bottom)
    ax = fig.axes[0]
    ax.set_xticks(df_years["Year"])
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    c.drawImage(
        ImageReader(img_buf),
        left_x,
        40,
        width=content_width,
        height=170,
        preserveAspectRatio=True,
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- PDF: MULTI-COMPARISON ----------------

def generate_multi_pdf(selected_rows, fig, comparison_df_num):
    """
    comparison_df_num: index=Customer, columns=years + 'Total'
    """
    buffer = io.BytesIO()

    page = landscape(letter)
    width, height = page
    c = canvas.Canvas(buffer, pagesize=page)

    # HEADER
    c.setFont("Helvetica-Bold", 26)
    c.setFillColor(REGAL_BLUE)
    c.drawCentredString(width / 2, height - 45, "Regal Plastics")

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 75, "Customer Comparison Report")

    c.setFillColor(colors.black)

    left_x = 40
    right_margin = 40
    content_width = width - left_x - right_margin
    y = height - 115

    # SUMMARY OVERVIEW
    c.setFont("Helvetica-Bold", 13)
    c.drawString(left_x, y, "Summary Overview")
    y -= 20

    c.setFont("Helvetica", 11)
    for r in selected_rows:
        total = comparison_df_num.loc[r["Customer Name"], "Total"]
        line = f"{r['Customer Name']}  |  #{r['Cust. #']}  |  Total Sales: {format_money(total)}"
        c.drawString(left_x, y, line)
        y -= 16

    # KPI CARDS STACKED (blue theme, smaller)
    totals_sorted = comparison_df_num["Total"].sort_values(ascending=False)
    best_name = totals_sorted.index[0]
    best_val = totals_sorted.iloc[0]
    worst_name = totals_sorted.index[-1]
    worst_val = totals_sorted.iloc[-1]
    combined_total = comparison_df_num["Total"].sum()
    avg = combined_total / len(comparison_df_num)

    kpis = [
        ("Top Customer", f"{best_name} ({format_money(best_val)})"),
        ("Lowest Customer", f"{worst_name} ({format_money(worst_val)})"),
        ("Total of Selected", format_money(combined_total)),
        ("Avg per Customer", format_money(avg)),
    ]

    card_x = left_x
    card_w = content_width
    card_h = 26
    spacing = 4
    kpi_y = y - 10

    for title, value in kpis:
        c.setFillColor(REGAL_LIGHT_BLUE)
        c.rect(card_x, kpi_y - card_h, card_w, card_h, fill=1, stroke=0)

        c.setStrokeColor(REGAL_BLUE)
        c.rect(card_x, kpi_y - card_h, card_w, card_h, fill=0, stroke=1)

        c.setFillColor(REGAL_BLUE)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(card_x + 8, kpi_y - 10, title)

        c.setFont("Helvetica", 10)
        c.setFillColor(colors.black)
        c.drawString(card_x + 8, kpi_y - 21, value)

        kpi_y -= (card_h + spacing)

    # VERTICAL SALES COMPARISON TABLE (years as rows, customers as columns)
    table_top_y = kpi_y - 30
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(REGAL_BLUE)
    c.drawString(left_x, table_top_y, "Sales Comparison Table")
    c.setFillColor(colors.black)
    table_y = table_top_y - 18

    # Restrict to at most 3 customers (you chose "1" = up to 3)
    customers = list(comparison_df_num.index)[:3]

    # Build vertical structure
    row_labels = YEAR_COLS + ["Total"]

    # Column positions
    year_col_width = 70
    remaining_width = content_width - year_col_width
    if len(customers) > 0:
        cust_col_width = remaining_width / len(customers)
    else:
        cust_col_width = remaining_width

    col_x = [left_x]
    col_x.append(left_x + year_col_width)
    for i in range(len(customers) - 1):
        col_x.append(col_x[-1] + cust_col_width)

    # Header row
    header_height = 18
    c.setFillColorRGB(0.88, 0.88, 0.88)
    table_total_width = year_col_width + cust_col_width * max(1, len(customers))
    c.rect(left_x, table_y - 4, table_total_width, header_height, fill=1, stroke=0)
    c.setFillColor(colors.black)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(col_x[0] + 2, table_y, "Year")

    for i, cust in enumerate(customers):
        label = cust if len(cust) <= 18 else cust[:16] + "…"
        c.drawString(col_x[i + 1] + 4, table_y, label)

    table_y -= header_height

    # Table rows (years)
    c.setFont("Helvetica", 9)
    row_h = 16

    for r_index, label in enumerate(row_labels):
        # alternate shading
        if r_index % 2 == 0:
            c.setFillColorRGB(0.96, 0.96, 0.96)
            c.rect(left_x, table_y - 2, table_total_width, row_h, fill=1, stroke=0)
            c.setFillColor(colors.black)

        c.drawString(col_x[0] + 2, table_y, label)

        for i, cust in enumerate(customers):
            if label == "Total":
                val = comparison_df_num.loc[cust, "Total"]
            else:
                year_int = int(label)
                val = comparison_df_num.loc[cust, year_int] if year_int in comparison_df_num.columns else np.nan
            s = format_money(val) if not pd.isna(val) else ""
            c.drawRightString(col_x[i + 1] + cust_col_width - 6, table_y, s)

        table_y -= row_h

    # CHART – full width bottom
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png", bbox_inches="tight")
    img_buf.seek(0)

    c.drawImage(
        ImageReader(img_buf),
        left_x,
        40,
        width=content_width,
        height=170,
        preserveAspectRatio=True,
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------------- STREAMLIT APP ----------------

st.set_page_config(page_title="Customer Sales Analytics", layout="wide")
st.title("📈 Customer Sales Analytics")

CSV_URL = "https://raw.githubusercontent.com/SystemAdmin-RP/customer-sales-analytics/main/CustomerTrend.csv"

admin_pass = st.sidebar.text_input("Admin Password", type="password")
uploaded = None

if admin_pass == "admin123":
    uploaded = st.sidebar.file_uploader("Upload CustomerTrend CSV", type="csv")

df = safe_read_csv(uploaded) if uploaded else safe_read_csv(CSV_URL)

# Parse numeric sales columns
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
            select_label = (
                results["Customer Name"]
                + " (#"
                + results["Cust. #"].astype(str)
                + ")"
            )

            selected = st.selectbox("Select a customer:", select_label)

            row = results.iloc[select_label.tolist().index(selected)]

            df_years = build_yearly_table(row)
            fig = create_trend_figure(df_years, row["Customer Name"])
            st.pyplot(fig)

            st.dataframe(
                df_years.style.format({
                    "Sales": format_money,
                    "YoY $ Change": format_money,
                    "YoY % Change": format_pct,
                }),
                hide_index=True,
            )

            pdf = generate_pdf_report(row, df_years, fig)
            st.download_button(
                "Download PDF Report",
                data=pdf,
                file_name=f"{row['Cust. #']}_report.pdf",
                mime="application/pdf",
            )


# ---------------- TAB 2: COMPARISON VIEW ----------------

with tab2:
    st.subheader("Compare multiple customers (up to 3 for PDF layout)")

    all_options = df["Customer Name"].astype(str) + " (#" + df["Cust. #"].astype(str) + ")"

    selected_customers = st.multiselect("Select 2–3 customers:", all_options)

    if len(selected_customers) < 2:
        st.info("Please choose at least two customers.")
    elif len(selected_customers) > 3:
        st.warning("For the PDF layout, please select no more than 3 customers.")
    else:
        selected_rows = []
        for name in selected_customers:
            r = df.iloc[(all_options == name).to_numpy().nonzero()[0][0]]
            selected_rows.append(r)

        fig2, ax = plt.subplots(figsize=(6, 3))

        comparison_rows = []
        for r in selected_rows:
            dfy = build_yearly_table(r)
            ax.plot(dfy["Year"], dfy["Sales"], marker="o", label=r["Customer Name"])

            row_data = {int(y): v for y, v in zip(dfy["Year"], dfy["Sales"])}
            row_data["Customer"] = r["Customer Name"]
            row_data["Total"] = dfy["Sales"].sum()
            comparison_rows.append(row_data)

        # X-axis years
        all_years = sorted({int(y) for y in YEAR_COLS})
        ax.set_xticks(all_years)
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.set_title("Customer Comparison Trend")
        ax.set_xlabel("Year")
        ax.set_ylabel("Sales")
        ax.grid(True)
        ax.legend()

        st.pyplot(fig2)

        comparison_df_num = pd.DataFrame(comparison_rows).set_index("Customer")
        st.dataframe(
            comparison_df_num.applymap(
                lambda v: format_money(v) if not pd.isna(v) else ""
            )
        )

        totals_sorted = comparison_df_num["Total"].sort_values(ascending=False)
        st.write(f"🏆 Best Customer: {totals_sorted.index[0]} — {format_money(totals_sorted.iloc[0])}")
        st.write(f"📉 Lowest Customer: {totals_sorted.index[-1]} — {format_money(totals_sorted.iloc[-1])}")

        pdf_multi = generate_multi_pdf(selected_rows, fig2, comparison_df_num)
        st.download_button(
            "Download Comparison PDF",
            data=pdf_multi,
            file_name="comparison_report.pdf",
            mime="application/pdf",
        )
