import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Valuation | DCF Calculator",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-box {
        background: #f0f4ff;
        border-left: 4px solid #4a6fa5;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 6px 0;
    }
    .formula-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 14px;
        font-family: monospace;
        font-size: 0.9em;
        margin: 8px 0;
    }
    .verdict-undervalued {
        background: #d4edda;
        border-left: 5px solid #28a745;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 1.1em;
    }
    .verdict-overvalued {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 1.1em;
    }
    .verdict-fair {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 12px 18px;
        border-radius: 6px;
        font-size: 1.1em;
    }
    .step-header {
        font-size: 1.05em;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 8px;
    }
    .info-callout {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str):
    """Pull all the data we need from yfinance in one shot."""
    tk = yf.Ticker(ticker)
    info = tk.info
    cf   = tk.cashflow          # annual, most-recent first
    inc  = tk.income_stmt
    bal  = tk.balance_sheet
    hist = tk.history(period="2y")
    return info, cf, inc, bal, hist


def safe_get(df, row_label, col_index=0, default=None):
    """Pull a value from a DataFrame by row label + column position, safely."""
    try:
        row = df.loc[row_label]
        val = row.iloc[col_index]
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def format_large(n):
    if n is None:
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


def compute_wacc(equity_weight, debt_weight, cost_of_equity,
                 cost_of_debt, tax_rate):
    """WACC = E/V * Re + D/V * Rd * (1 - T)"""
    return (equity_weight * cost_of_equity +
            debt_weight * cost_of_debt * (1 - tax_rate))


def compute_capm(risk_free, beta, equity_risk_premium):
    """CAPM: Re = Rf + β × ERP"""
    return risk_free + beta * equity_risk_premium


def run_dcf(base_fcf, growth_rates_stage1, terminal_growth,
            wacc, shares_outstanding, net_debt):
    """
    Two-stage DCF:
      Stage 1 – explicit FCF projections for each year in growth_rates_stage1
      Stage 2 – Gordon Growth terminal value
    Returns intrinsic value per share and the full cash-flow table.
    """
    rows = []
    pv_sum = 0.0
    fcf = base_fcf

    for yr, g in enumerate(growth_rates_stage1, start=1):
        fcf = fcf * (1 + g)
        discount_factor = 1 / (1 + wacc) ** yr
        pv = fcf * discount_factor
        pv_sum += pv
        rows.append({
            "Year": yr,
            "Growth Rate": f"{g*100:.1f}%",
            "FCF ($M)": round(fcf / 1e6, 1),
            "Discount Factor": round(discount_factor, 4),
            "PV of FCF ($M)": round(pv / 1e6, 1),
        })

    # Terminal value (Gordon Growth)
    terminal_fcf = fcf * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    tv_pv = terminal_value / (1 + wacc) ** len(growth_rates_stage1)
    pv_sum += tv_pv

    enterprise_value = pv_sum
    equity_value = enterprise_value - net_debt
    intrinsic_per_share = equity_value / shares_outstanding if shares_outstanding else 0

    return {
        "rows": rows,
        "terminal_value": terminal_value,
        "tv_pv": tv_pv,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_per_share": intrinsic_per_share,
        "pv_sum_stage1": pv_sum - tv_pv,
    }


# ── App starts here ───────────────────────────────────────────────────────────

st.title("📊 Equity Intrinsic Value Calculator")
st.markdown("**Two-Stage DCF Model** | FINA 4011/5011 · Project 2")

st.markdown("""
<div class="info-callout">
This app estimates a stock's <b>intrinsic value per share</b> using a two-stage 
Discounted Cash Flow (DCF) model. You supply the assumptions; the model retrieves 
live financial data and walks you through every calculation step.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Step 0 – Ticker Input ─────────────────────────────────────────────────────
st.header("Step 1 · Select a Stock")
col_tick, col_btn = st.columns([2, 1])

with col_tick:
    ticker_input = st.text_input(
        "Enter stock ticker (e.g., AAPL, MSFT, TSLA)",
        value="AAPL",
        help="Must be a valid US equity ticker listed on Yahoo Finance."
    ).upper().strip()

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    load_btn = st.button("🔄 Load Data", use_container_width=True)

if "ticker" not in st.session_state:
    st.session_state.ticker = ticker_input

if load_btn:
    st.session_state.ticker = ticker_input

ticker = st.session_state.ticker

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**…"):
    try:
        info, cf, inc, bal, hist = fetch_stock_data(ticker)
    except Exception as e:
        st.error(f"Could not load data for '{ticker}'. Check the ticker and try again.\n\n{e}")
        st.stop()

if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
    st.error("Ticker not found or no price data available. Please try another ticker.")
    st.stop()

# ── Step 1 – Company Snapshot ─────────────────────────────────────────────────
company_name   = info.get("longName", ticker)
current_price  = info.get("currentPrice") or info.get("regularMarketPrice", 0)
market_cap     = info.get("marketCap", 0)
sector         = info.get("sector", "N/A")
industry       = info.get("industry", "N/A")
beta_live      = info.get("beta", 1.0) or 1.0
shares_out     = info.get("sharesOutstanding", 1)
pe_ratio       = info.get("trailingPE")
fwd_pe         = info.get("forwardPE")
ev_ebitda      = info.get("enterpriseToEbitda")
div_yield      = info.get("dividendYield", 0) or 0
description    = info.get("longBusinessSummary", "")

st.subheader(f"🏢 {company_name} ({ticker})")
st.caption(f"{sector} · {industry}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"${current_price:.2f}")
c2.metric("Market Cap",     format_large(market_cap))
c3.metric("Beta (live)",    f"{beta_live:.2f}")
c4.metric("Trailing P/E",   f"{pe_ratio:.1f}x" if pe_ratio else "N/A")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Forward P/E",    f"{fwd_pe:.1f}x" if fwd_pe else "N/A")
c6.metric("EV/EBITDA",      f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A")
c7.metric("Dividend Yield", f"{div_yield*100:.2f}%")
c8.metric("Shares Out",     format_large(shares_out).replace("$", ""))

if description:
    with st.expander("📋 Business Description"):
        st.write(description)

# Historical price chart
if not hist.empty:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name="Close Price",
        line=dict(color="#4a6fa5", width=2)
    ))
    fig_hist.update_layout(
        title=f"{ticker} – 2-Year Price History",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=280, margin=dict(t=40, b=20, l=20, r=20),
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Step 2 – Pull raw financials ──────────────────────────────────────────────
st.header("Step 2 · Financial Data Retrieved Automatically")

# FCF = Operating CF – CapEx
try:
    op_cf_vals = []
    capex_vals  = []
    fcf_vals    = []
    years       = []

    for i in range(min(4, cf.shape[1])):
        ocf = safe_get(cf, "Operating Cash Flow", i)
        capex = safe_get(cf, "Capital Expenditure", i)
        if ocf is not None and capex is not None:
            fcf = ocf + capex   # capex is negative in yfinance
            op_cf_vals.append(ocf / 1e6)
            capex_vals.append(capex / 1e6)
            fcf_vals.append(fcf / 1e6)
            years.append(cf.columns[i].year)

    base_fcf = fcf_vals[0] * 1e6 if fcf_vals else 0.0

    # Revenue, Net Income
    rev_vals = [safe_get(inc, "Total Revenue", i) for i in range(min(4, inc.shape[1]))]
    ni_vals  = [safe_get(inc, "Net Income", i)    for i in range(min(4, inc.shape[1]))]
    rev_vals = [v/1e6 if v else None for v in rev_vals]
    ni_vals  = [v/1e6 if v else None for v in ni_vals]
    inc_years = [inc.columns[i].year for i in range(min(4, inc.shape[1]))]

    # Debt & cash for net debt
    total_debt = safe_get(bal, "Total Debt", 0, 0)
    cash       = safe_get(bal, "Cash And Cash Equivalents", 0, 0)
    net_debt   = (total_debt - cash) if (total_debt and cash) else 0.0

    data_ok = bool(fcf_vals)
except Exception as ex:
    st.warning(f"Some financial data could not be parsed: {ex}")
    base_fcf = 0.0
    net_debt = 0.0
    data_ok  = False

if data_ok:
    col_left, col_right = st.columns(2)

    with col_left:
        # FCF bar chart
        fcf_df = pd.DataFrame({
            "Year": years,
            "Operating CF": op_cf_vals,
            "CapEx": capex_vals,
            "Free Cash Flow": fcf_vals
        }).sort_values("Year")

        fig_cf = go.Figure()
        fig_cf.add_bar(x=fcf_df["Year"], y=fcf_df["Operating CF"],
                       name="Operating CF", marker_color="#4a6fa5")
        fig_cf.add_bar(x=fcf_df["Year"], y=fcf_df["CapEx"],
                       name="CapEx", marker_color="#e07b54")
        fig_cf.add_scatter(x=fcf_df["Year"], y=fcf_df["Free Cash Flow"],
                           mode="lines+markers", name="FCF",
                           line=dict(color="#2ca02c", width=2, dash="dot"),
                           marker=dict(size=8))
        fig_cf.update_layout(
            title="Cash Flow Breakdown ($M)", barmode="group",
            height=320, template="plotly_white",
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_cf, use_container_width=True)

    with col_right:
        # Revenue & Net Income
        inc_df = pd.DataFrame({
            "Year": inc_years,
            "Revenue": [v for v in rev_vals],
            "Net Income": [v for v in ni_vals]
        }).dropna().sort_values("Year")

        fig_inc = go.Figure()
        fig_inc.add_bar(x=inc_df["Year"], y=inc_df["Revenue"],
                        name="Revenue", marker_color="#5b9bd5")
        fig_inc.add_bar(x=inc_df["Year"], y=inc_df["Net Income"],
                        name="Net Income", marker_color="#70ad47")
        fig_inc.update_layout(
            title="Revenue & Net Income ($M)", barmode="group",
            height=320, template="plotly_white",
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown(f"""
    <div class="metric-box">
    📌 <b>Base Free Cash Flow (most recent year):</b> {format_large(base_fcf)}&emsp;
    | &emsp;<b>Net Debt:</b> {format_large(net_debt)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
    <b>How FCF is calculated:</b><br>
    Free Cash Flow = Operating Cash Flow − Capital Expenditures<br>
    This is the cash the company generates after maintaining/expanding its asset base.
    It represents what's available to repay debt, pay dividends, or reinvest.
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Could not retrieve complete cash flow data. "
               "You can still proceed by entering FCF manually below.")
    base_fcf = 0.0
    net_debt = 0.0

st.divider()

# ── Step 3 – WACC Inputs ──────────────────────────────────────────────────────
st.header("Step 3 · Cost of Capital (WACC)")

st.markdown("""
<div class="info-callout">
<b>WACC</b> (Weighted Average Cost of Capital) is the discount rate applied to 
future cash flows. It reflects the blended required return of all capital providers 
(equity holders and debt holders). A higher WACC means a lower valuation.
</div>
""", unsafe_allow_html=True)

with st.expander("📐 WACC Formula"):
    st.markdown("""
    <div class="formula-box">
    WACC = (E/V) × Re + (D/V) × Rd × (1 − Tax Rate)<br><br>
    where:<br>
    &nbsp;&nbsp;E = Market value of equity (Market Cap)<br>
    &nbsp;&nbsp;D = Market value of debt<br>
    &nbsp;&nbsp;V = E + D<br>
    &nbsp;&nbsp;Re = Cost of equity (CAPM)<br>
    &nbsp;&nbsp;Rd = Pre-tax cost of debt<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **Cost of Equity via CAPM:**
    <div class="formula-box">
    Re = Rf + β × (Rm − Rf)<br><br>
    where:<br>
    &nbsp;&nbsp;Rf = Risk-free rate (10-yr Treasury yield)<br>
    &nbsp;&nbsp;β  = Stock's sensitivity to market movements<br>
    &nbsp;&nbsp;(Rm − Rf) = Equity Risk Premium<br>
    </div>
    """, unsafe_allow_html=True)

col_w1, col_w2 = st.columns(2)

with col_w1:
    st.markdown("#### CAPM Inputs")
    risk_free = st.slider(
        "Risk-Free Rate (Rf) – 10-yr Treasury Yield (%)",
        min_value=0.5, max_value=8.0, value=4.3, step=0.1,
        help="As of early 2025, the 10-yr US Treasury yield is ~4.2–4.5%. "
             "This represents the return on a 'risk-free' investment."
    ) / 100

    erp = st.slider(
        "Equity Risk Premium (ERP) (%)",
        min_value=2.0, max_value=8.0, value=5.5, step=0.1,
        help="Historical US ERP is ~5–6%. This is the extra return investors "
             "demand for holding equities over risk-free assets."
    ) / 100

    beta = st.number_input(
        f"Beta (β) — live value: {beta_live:.2f}",
        min_value=0.1, max_value=3.0, value=round(beta_live, 2), step=0.05,
        help="Beta measures stock volatility relative to the market. "
             "β>1 = more volatile than market; β<1 = less volatile."
    )

    cost_of_equity = compute_capm(risk_free, beta, erp)
    st.success(f"**Cost of Equity (CAPM): {cost_of_equity*100:.2f}%**")
    st.caption(f"= {risk_free*100:.1f}% + {beta:.2f} × {erp*100:.1f}%")

with col_w2:
    st.markdown("#### Capital Structure Inputs")

    equity_pct = st.slider(
        "Equity Weight (E/V) (%)",
        min_value=10, max_value=100, value=80, step=5,
        help="The proportion of total capital financed by equity. "
             "Higher equity weight = less financial leverage."
    )
    debt_pct = 100 - equity_pct
    st.caption(f"⟹ Debt Weight (D/V) = {debt_pct}%")

    cost_of_debt = st.slider(
        "Pre-Tax Cost of Debt (Rd) (%)",
        min_value=1.0, max_value=12.0, value=4.0, step=0.25,
        help="The yield on the company's outstanding debt. "
             "Check the company's 10-K or Bloomberg for exact figures."
    ) / 100

    tax_rate = st.slider(
        "Effective Tax Rate (%)",
        min_value=5, max_value=40, value=21, step=1,
        help="US corporate tax rate is 21%. Effective rates vary by company."
    ) / 100

    wacc = compute_wacc(
        equity_pct / 100, debt_pct / 100,
        cost_of_equity, cost_of_debt, tax_rate
    )
    st.success(f"**WACC: {wacc*100:.2f}%**")

# WACC breakdown gauge
fig_wacc = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(wacc * 100, 2),
    title={"text": "WACC (%)"},
    gauge={
        "axis": {"range": [0, 20]},
        "bar": {"color": "#4a6fa5"},
        "steps": [
            {"range": [0, 6],   "color": "#d4edda"},
            {"range": [6, 12],  "color": "#fff3cd"},
            {"range": [12, 20], "color": "#f8d7da"},
        ],
        "threshold": {
            "line": {"color": "red", "width": 2},
            "thickness": 0.75,
            "value": wacc * 100,
        },
    },
    number={"suffix": "%", "font": {"size": 36}},
))
fig_wacc.update_layout(height=240, margin=dict(t=20, b=10))
st.plotly_chart(fig_wacc, use_container_width=True)

st.divider()

# ── Step 4 – Growth Assumptions ───────────────────────────────────────────────
st.header("Step 4 · FCF Growth Assumptions")

st.markdown("""
<div class="info-callout">
This model uses a <b>two-stage approach</b>: an explicit high-growth period 
(Years 1–10) followed by a stable perpetuity (terminal) phase. You can set 
different growth rates for each year in Stage 1 to reflect how the business 
is expected to evolve.
</div>
""", unsafe_allow_html=True)

col_g1, col_g2 = st.columns([2, 1])

with col_g1:
    st.markdown("#### Stage 1 – Explicit Growth Period (Years 1–10)")

    analyst_growth = info.get("earningsGrowth") or info.get("revenueGrowth") or None
    if analyst_growth:
        st.info(f"💡 Analyst consensus growth estimate for {ticker}: **{analyst_growth*100:.1f}%** "
                f"(use as a reference, not gospel).")

    forecast_years = st.slider(
        "Number of explicit forecast years",
        min_value=3, max_value=10, value=5, step=1,
        help="More years = more detailed model. 5–10 years is standard."
    )

    use_custom = st.toggle("Set custom growth rate per year", value=False)
    growth_rates = []

    if use_custom:
        cols = st.columns(forecast_years)
        for i, c in enumerate(cols):
            g = c.number_input(
                f"Yr {i+1} (%)", min_value=-30.0, max_value=50.0,
                value=15.0 if i < 3 else 10.0,
                step=0.5, key=f"gr_{i}"
            )
            growth_rates.append(g / 100)
    else:
        g_high = st.slider(
            "Near-term growth rate (Years 1–3) (%)",
            min_value=-10.0, max_value=40.0, value=15.0, step=0.5
        ) / 100
        g_mid = st.slider(
            "Mid-term growth rate (Years 4–end) (%)",
            min_value=-10.0, max_value=30.0, value=10.0, step=0.5
        ) / 100
        growth_rates = [g_high] * min(3, forecast_years) + \
                       [g_mid] * max(0, forecast_years - 3)

with col_g2:
    st.markdown("#### Stage 2 – Terminal Growth")
    terminal_growth = st.slider(
        "Perpetuity Growth Rate (%)",
        min_value=0.0, max_value=5.0, value=2.5, step=0.1,
        help="Long-run sustainable growth — typically close to nominal GDP growth "
             "(2–3%). MUST be less than WACC."
    ) / 100

    if terminal_growth >= wacc:
        st.error("⚠️ Terminal growth rate must be below WACC! "
                 "Reduce terminal growth or increase WACC.")

    st.markdown("""
    <div class="info-callout">
    <b>Terminal Value</b> captures all cash flows beyond the explicit forecast 
    period using the Gordon Growth Model:<br><br>
    TV = FCF_n × (1 + g) / (WACC − g)
    </div>
    """, unsafe_allow_html=True)

# Base FCF override
st.markdown("#### Base FCF Override")
base_fcf_input = st.number_input(
    f"Base Free Cash Flow ($M) — auto-pulled: {base_fcf/1e6:.1f}M",
    min_value=-50000.0, max_value=500000.0,
    value=round(base_fcf / 1e6, 1), step=50.0,
    help="Starting FCF for the projection. Defaults to the most recent reported FCF. "
         "Override if you want to normalize for one-time items."
) * 1e6

st.divider()

# ── Step 5 – Run the DCF ──────────────────────────────────────────────────────
st.header("Step 5 · DCF Output & Valuation")

if terminal_growth >= wacc:
    st.error("Fix the terminal growth rate before running the model.")
    st.stop()

if base_fcf_input == 0:
    st.warning("Base FCF is zero — results may not be meaningful. "
               "Check the financial data or enter a manual override.")

result = run_dcf(
    base_fcf=base_fcf_input,
    growth_rates_stage1=growth_rates,
    terminal_growth=terminal_growth,
    wacc=wacc,
    shares_outstanding=shares_out,
    net_debt=net_debt,
)

# ── DCF table ─────────────────────────────────────────────────────────────────
st.markdown("### 📋 Year-by-Year Cash Flow Projections")

dcf_df = pd.DataFrame(result["rows"])
st.dataframe(
    dcf_df.style.format({
        "FCF ($M)": "{:,.1f}",
        "PV of FCF ($M)": "{:,.1f}",
        "Discount Factor": "{:.4f}",
    }),
    use_container_width=True,
    hide_index=True
)

# Summary numbers
pv_stage1  = result["pv_sum_stage1"]
tv         = result["terminal_value"]
tv_pv      = result["tv_pv"]
ev         = result["enterprise_value"]
eq_val     = result["equity_value"]
iv_ps      = result["intrinsic_per_share"]

c1, c2, c3 = st.columns(3)
c1.metric("PV of Stage 1 FCFs", format_large(pv_stage1))
c2.metric("Terminal Value (undiscounted)", format_large(tv))
c3.metric("PV of Terminal Value", format_large(tv_pv))

c4, c5, c6 = st.columns(3)
c4.metric("Enterprise Value", format_large(ev))
c5.metric("(−) Net Debt", format_large(net_debt))
c6.metric("Equity Value", format_large(eq_val))

st.markdown("---")
st.subheader("🎯 Intrinsic Value Per Share")

margin_of_safety = (iv_ps - current_price) / iv_ps * 100 if iv_ps != 0 else 0
upside_downside  = (iv_ps - current_price) / current_price * 100

col_val, col_gauge = st.columns([1, 1])

with col_val:
    st.metric(
        label="Intrinsic Value (DCF)",
        value=f"${iv_ps:.2f}",
        delta=f"{upside_downside:+.1f}% vs. market price ${current_price:.2f}"
    )

    if iv_ps > current_price * 1.15:
        st.markdown(f"""
        <div class="verdict-undervalued">
        ✅ <b>POTENTIALLY UNDERVALUED</b><br>
        The stock appears to trade at a <b>{abs(upside_downside):.1f}% discount</b> 
        to intrinsic value. Margin of safety: <b>{margin_of_safety:.1f}%</b>.
        </div>
        """, unsafe_allow_html=True)
    elif iv_ps < current_price * 0.85:
        st.markdown(f"""
        <div class="verdict-overvalued">
        ❌ <b>POTENTIALLY OVERVALUED</b><br>
        The stock appears to trade at a <b>{abs(upside_downside):.1f}% premium</b> 
        to intrinsic value. The market may already be pricing in strong growth.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-fair">
        ⚖️ <b>APPROXIMATELY FAIRLY VALUED</b><br>
        The stock trades within 15% of intrinsic value. 
        Sensitivity analysis below will help clarify the range.
        </div>
        """, unsafe_allow_html=True)

with col_gauge:
    # Value composition waterfall
    labels = ["PV Stage 1 FCFs", "PV Terminal Value",
              "Enterprise Value", "Less: Net Debt", "Equity Value"]
    values = [pv_stage1 / 1e9, tv_pv / 1e9,
              ev / 1e9, -net_debt / 1e9, eq_val / 1e9]
    fig_wf = go.Figure(go.Waterfall(
        name="Valuation",
        orientation="v",
        x=labels,
        y=[pv_stage1 / 1e9, tv_pv / 1e9, 0, -net_debt / 1e9, 0],
        measure=["relative", "relative", "total", "relative", "total"],
        text=[f"${v:.1f}B" for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#e07b54"}},
        increasing={"marker": {"color": "#4a6fa5"}},
        totals={"marker": {"color": "#2ca02c"}},
    ))
    fig_wf.update_layout(
        title="Valuation Build-Up ($B)",
        height=350, template="plotly_white",
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_wf, use_container_width=True)

st.divider()

# ── Step 6 – Sensitivity Analysis ─────────────────────────────────────────────
st.header("Step 6 · Sensitivity Analysis")

st.markdown("""
<div class="info-callout">
DCF models are highly sensitive to input assumptions. This heat map shows how 
intrinsic value per share changes across a range of WACC and terminal growth rates. 
<b>Green = above current market price; Red = below current market price.</b>
</div>
""", unsafe_allow_html=True)

wacc_range = np.arange(wacc - 0.03, wacc + 0.035, 0.005)
tg_range   = np.arange(max(0.005, terminal_growth - 0.015),
                        min(wacc - 0.01, terminal_growth + 0.02),
                        0.005)

sensitivity_matrix = []
for tg in tg_range:
    row = []
    for w in wacc_range:
        if tg >= w:
            row.append(np.nan)
        else:
            r = run_dcf(base_fcf_input, growth_rates, tg, w, shares_out, net_debt)
            row.append(round(r["intrinsic_per_share"], 2))
    sensitivity_matrix.append(row)

sens_df = pd.DataFrame(
    sensitivity_matrix,
    index=[f"{tg*100:.1f}%" for tg in tg_range],
    columns=[f"{w*100:.1f}%" for w in wacc_range],
)

# Color: green if IV > current price, red otherwise
def color_cells(val):
    if pd.isna(val):
        return "background-color: #cccccc"
    if val > current_price * 1.15:
        intensity = min(255, int(200 - (val / current_price - 1) * 80))
        return f"background-color: rgb({intensity},210,{intensity}); color: black"
    elif val < current_price * 0.85:
        intensity = min(255, int(200 + (1 - val / current_price) * 80))
        return f"background-color: rgb(210,{intensity},{intensity}); color: black"
    else:
        return "background-color: #fff3cd; color: black"

st.markdown("**Intrinsic Value Per Share — by WACC (columns) × Terminal Growth (rows)**")
styled = sens_df.style.applymap(color_cells).format("${:.2f}", na_rep="N/A")
st.dataframe(styled, use_container_width=True)

st.divider()

# ── Step 7 – Scenario Analysis ────────────────────────────────────────────────
st.header("Step 7 · Bear / Base / Bull Scenario Comparison")

st.markdown("""
<div class="info-callout">
Three scenarios stress-test the valuation. Bear = pessimistic; Base = your current 
inputs; Bull = optimistic. Each scenario uses different growth and discount assumptions.
</div>
""", unsafe_allow_html=True)

sc_col1, sc_col2, sc_col3 = st.columns(3)

with sc_col1:
    st.markdown("🐻 **Bear Case**")
    bear_growth = st.slider("Bear: near-term growth (%)", -10, 30,
                             max(0, int(growth_rates[0] * 100 - 8)),
                             key="bear_g") / 100
    bear_wacc   = st.slider("Bear: WACC (%)", 5, 20,
                             min(20, int(wacc * 100 + 2)),
                             key="bear_w") / 100
    bear_tg     = st.slider("Bear: terminal growth (%)", 0, 4, 1,
                             key="bear_tg") / 100

with sc_col2:
    st.markdown("⚖️ **Base Case** (your inputs)")
    st.metric("Near-term growth", f"{growth_rates[0]*100:.1f}%")
    st.metric("WACC", f"{wacc*100:.2f}%")
    st.metric("Terminal growth", f"{terminal_growth*100:.1f}%")

with sc_col3:
    st.markdown("🐂 **Bull Case**")
    bull_growth = st.slider("Bull: near-term growth (%)", 0, 60,
                             min(50, int(growth_rates[0] * 100 + 8)),
                             key="bull_g") / 100
    bull_wacc   = st.slider("Bull: WACC (%)", 4, 18,
                             max(4, int(wacc * 100 - 2)),
                             key="bull_w") / 100
    bull_tg     = st.slider("Bull: terminal growth (%)", 1, 5, 3,
                             key="bull_tg") / 100

# Compute scenarios
def scenario_rates(base, override, n):
    return [override] * min(3, n) + [max(0, override * 0.7)] * max(0, n - 3)

bear_r = run_dcf(base_fcf_input,
                 scenario_rates(growth_rates, bear_growth, forecast_years),
                 bear_tg, bear_wacc, shares_out, net_debt)
bull_r = run_dcf(base_fcf_input,
                 scenario_rates(growth_rates, bull_growth, forecast_years),
                 bull_tg, bull_wacc, shares_out, net_debt)
base_r = result

scen_df = pd.DataFrame({
    "Scenario":   ["🐻 Bear", "⚖️ Base", "🐂 Bull"],
    "Intrinsic Value": [bear_r["intrinsic_per_share"],
                         base_r["intrinsic_per_share"],
                         bull_r["intrinsic_per_share"]],
    "Upside vs. Market": [
        f"{(bear_r['intrinsic_per_share'] - current_price)/current_price*100:+.1f}%",
        f"{(base_r['intrinsic_per_share'] - current_price)/current_price*100:+.1f}%",
        f"{(bull_r['intrinsic_per_share'] - current_price)/current_price*100:+.1f}%",
    ],
    "EV ($B)": [round(v["enterprise_value"] / 1e9, 1) for v in
                [bear_r, base_r, bull_r]],
})

fig_scen = go.Figure()
colors = ["#e07b54", "#4a6fa5", "#2ca02c"]
for i, (_, row) in enumerate(scen_df.iterrows()):
    fig_scen.add_bar(
        x=[row["Scenario"]],
        y=[row["Intrinsic Value"]],
        name=row["Scenario"],
        marker_color=colors[i],
        text=[f"${row['Intrinsic Value']:.2f}"],
        textposition="outside",
    )
fig_scen.add_hline(
    y=current_price,
    line_dash="dash", line_color="black",
    annotation_text=f"Market Price ${current_price:.2f}",
    annotation_position="top right"
)
fig_scen.update_layout(
    title="Intrinsic Value per Share — Scenarios",
    yaxis_title="$/share",
    height=350, template="plotly_white",
    showlegend=False,
    margin=dict(t=50, b=20, l=20, r=20)
)
st.plotly_chart(fig_scen, use_container_width=True)
st.dataframe(scen_df.set_index("Scenario").style.format({"Intrinsic Value": "${:.2f}",
                                                           "EV ($B)": "{:.1f}B"}),
             use_container_width=True)

st.divider()

# ── Step 8 – Comps / Multiples ────────────────────────────────────────────────
st.header("Step 8 · Relative Valuation Benchmarks")

st.markdown("""
<div class="info-callout">
DCF is an absolute valuation method. This section adds context by comparing 
key multiples to common benchmarks — not a replacement for DCF, but a useful 
sanity check.
</div>
""", unsafe_allow_html=True)

multiples_data = {
    "Metric":       ["Trailing P/E", "Forward P/E", "EV/EBITDA", "Price/Book"],
    f"{ticker}":   [
        f"{pe_ratio:.1f}x"   if pe_ratio   else "N/A",
        f"{fwd_pe:.1f}x"     if fwd_pe     else "N/A",
        f"{ev_ebitda:.1f}x"  if ev_ebitda  else "N/A",
        f"{info.get('priceToBook', 0):.1f}x" if info.get("priceToBook") else "N/A",
    ],
    "S&P 500 Avg":  ["~22x", "~20x", "~14x", "~4x"],
    "Interpretation": [
        "< 15x cheap; 15–25x fair; > 25x premium",
        "Forward-looking; lower is better",
        "< 10x cheap; 10–20x fair; > 20x premium",
        "< 1x deep value; > 5x growth premium",
    ],
}
st.dataframe(pd.DataFrame(multiples_data).set_index("Metric"),
             use_container_width=True)

st.divider()

# ── Step 9 – Model Explanation ────────────────────────────────────────────────
st.header("Step 9 · Full Step-by-Step Calculation Summary")

with st.expander("📖 Click to view the complete calculation walkthrough"):
    st.markdown(f"""
    #### 1. Base Free Cash Flow
    - Most recent Operating CF minus CapEx → **{format_large(base_fcf_input)}**
    - This is the starting point for all projections

    #### 2. Cost of Equity (CAPM)
    - Re = Rf + β × ERP
    - Re = **{risk_free*100:.1f}%** + **{beta:.2f}** × **{erp*100:.1f}%**
    - Re = **{cost_of_equity*100:.2f}%**

    #### 3. WACC
    - WACC = (E/V) × Re + (D/V) × Rd × (1 − T)
    - WACC = ({equity_pct}% × {cost_of_equity*100:.2f}%) + ({debt_pct}% × {cost_of_debt*100:.2f}% × (1 − {tax_rate*100:.0f}%))
    - WACC = **{wacc*100:.2f}%**

    #### 4. Stage 1 FCF Projections
    """)

    for r in result["rows"]:
        st.markdown(f"""
    - **Year {r['Year']}**: FCF grows at {r['Growth Rate']} → ${r['FCF ($M)']:,.1f}M  
      Discounted at factor {r['Discount Factor']:.4f} → PV = **${r['PV of FCF ($M)']:,.1f}M**
        """)

    st.markdown(f"""
    #### 5. Terminal Value
    - Last projected FCF × (1 + g) / (WACC − g)
    - = {format_large(result['rows'][-1]['FCF ($M)']*1e6)} × (1 + {terminal_growth*100:.1f}%) / ({wacc*100:.2f}% − {terminal_growth*100:.1f}%)
    - Terminal Value (undiscounted) = **{format_large(tv)}**
    - PV of Terminal Value = **{format_large(tv_pv)}**

    #### 6. Enterprise Value
    - PV Stage 1 + PV Terminal Value
    - = {format_large(pv_stage1)} + {format_large(tv_pv)}
    - = **{format_large(ev)}**

    #### 7. Equity Value & Intrinsic Value Per Share
    - Equity Value = Enterprise Value − Net Debt
    - = {format_large(ev)} − {format_large(net_debt)} = **{format_large(eq_val)}**
    - Intrinsic Value / Share = {format_large(eq_val)} ÷ {format_large(shares_out).replace('$', '')} shares
    - = **${iv_ps:.2f}**

    #### 8. Verdict
    - Current Market Price: **${current_price:.2f}**
    - Intrinsic Value:      **${iv_ps:.2f}**
    - Upside / (Downside):  **{upside_downside:+.1f}%**
    """)

st.divider()
st.caption("⚠️ Disclaimer: This app is for educational purposes only. "
           "Intrinsic value estimates are highly sensitive to assumptions "
           "and should not be used as sole investment advice.")
