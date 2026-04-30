import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
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
    .info-callout {
        background: #e8f4f8;
        border-left: 4px solid #17a2b8;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)


# ── Data fetching via Yahoo Finance JSON API ──────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

@st.cache_data(ttl=300, show_spinner=False)
def fetch_summary(ticker: str) -> dict:
    url = (
        f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
        "?modules=price,summaryDetail,defaultKeyStatistics,"
        "financialData,assetProfile,incomeStatementHistory,"
        "cashflowStatementHistory,balanceSheetHistory"
    )
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    data = r.json()
    result = data.get("quoteSummary", {}).get("result")
    if not result:
        raise ValueError("No data returned. Check the ticker.")
    return result[0]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str) -> pd.DataFrame:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        "?range=2y&interval=1d"
    )
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    chart = r.json()["chart"]["result"][0]
    timestamps = chart["timestamp"]
    closes = chart["indicators"]["quote"][0]["close"]
    df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(timestamps, unit="s"))
    return df.dropna()


def raw(d: dict, *keys, default=None):
    """Safely dig into nested dicts, returning the 'raw' numeric value."""
    try:
        for k in keys:
            d = d[k]
        return d.get("raw", default)
    except Exception:
        return default


def fmt(n):
    if n is None:
        return "N/A"
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    if abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    return f"${n:,.0f}"


# ── DCF Engine ────────────────────────────────────────────────────────────────

def compute_capm(rf, beta, erp):
    return rf + beta * erp


def compute_wacc(eq_w, debt_w, re, rd, tax):
    return eq_w * re + debt_w * rd * (1 - tax)


def run_dcf(base_fcf, growth_rates, terminal_growth, wacc, shares, net_debt):
    rows = []
    pv_sum = 0.0
    fcf = base_fcf
    for yr, g in enumerate(growth_rates, 1):
        fcf *= (1 + g)
        df_ = 1 / (1 + wacc) ** yr
        pv  = fcf * df_
        pv_sum += pv
        rows.append({
            "Year": yr,
            "Growth Rate": f"{g*100:.1f}%",
            "FCF ($M)": round(fcf / 1e6, 1),
            "Discount Factor": round(df_, 4),
            "PV of FCF ($M)": round(pv / 1e6, 1),
        })
    terminal_fcf = fcf * (1 + terminal_growth)
    tv           = terminal_fcf / (wacc - terminal_growth)
    tv_pv        = tv / (1 + wacc) ** len(growth_rates)
    pv_sum      += tv_pv
    ev           = pv_sum
    equity_val   = ev - net_debt
    iv_ps        = equity_val / shares if shares else 0
    return {
        "rows": rows,
        "terminal_value": tv,
        "tv_pv": tv_pv,
        "enterprise_value": ev,
        "equity_value": equity_val,
        "intrinsic_per_share": iv_ps,
        "pv_stage1": pv_sum - tv_pv,
    }


# ═════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

st.title("📊 Equity Intrinsic Value Calculator")
st.markdown("**Two-Stage DCF Model** · FINA 4011/5011 Project 2")
st.markdown("""
<div class="info-callout">
This app estimates a stock's <b>intrinsic value per share</b> using a two-stage
Discounted Cash Flow (DCF) model. Financial data is pulled live from Yahoo Finance.
Enter your assumptions and walk through each calculation step below.
</div>
""", unsafe_allow_html=True)
st.divider()

# ── Step 1 · Ticker ───────────────────────────────────────────────────────────
st.header("Step 1 · Select a Stock")
col_t, col_b = st.columns([2, 1])
with col_t:
    ticker_input = st.text_input("Enter stock ticker (e.g. AAPL, MSFT, TSLA)",
                                  value="AAPL").upper().strip()
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    load_btn = st.button("🔄 Load Data", use_container_width=True)

if "ticker" not in st.session_state:
    st.session_state.ticker = ticker_input
if load_btn:
    st.session_state.ticker = ticker_input
ticker = st.session_state.ticker

# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for **{ticker}**…"):
    try:
        data = fetch_summary(ticker)
        hist = fetch_history(ticker)
    except Exception as e:
        st.error(f"Could not load data for '{ticker}'. Check the ticker and try again.\n\n{e}")
        st.stop()

# Parse summary modules
price_mod  = data.get("price", {})
fin_mod    = data.get("financialData", {})
key_mod    = data.get("defaultKeyStatistics", {})
profile    = data.get("assetProfile", {})
cf_stmts   = data.get("cashflowStatementHistory", {}).get("cashflowStatements", [])
inc_stmts  = data.get("incomeStatementHistory",   {}).get("incomeStatements",   [])
bal_stmts  = data.get("balanceSheetHistory",       {}).get("balanceSheetStatements", [])

company_name  = price_mod.get("longName", ticker)
current_price = raw(price_mod, "regularMarketPrice") or 0
market_cap    = raw(price_mod, "marketCap") or 0
beta_live     = raw(key_mod,   "beta") or 1.0
shares_out    = raw(key_mod,   "sharesOutstanding") or 1
pe_ratio      = raw(key_mod,   "trailingPE")
fwd_pe        = raw(key_mod,   "forwardPE")
ev_ebitda     = raw(key_mod,   "enterpriseToEbitda")
div_yield     = raw(key_mod,   "dividendYield") or 0
sector        = profile.get("sector", "N/A")
industry      = profile.get("industry", "N/A")
description   = profile.get("longBusinessSummary", "")

# Pull cash flow data
fcf_vals, years = [], []
op_cf_vals, capex_vals = [], []
for stmt in cf_stmts[:4]:
    ocf   = raw(stmt, "totalCashFromOperatingActivities")
    capex = raw(stmt, "capitalExpenditures")
    yr    = stmt.get("endDate", {}).get("fmt", "")[:4]
    if ocf is not None and capex is not None:
        fcf = ocf + capex   # capex is negative in Yahoo Finance
        op_cf_vals.append(ocf / 1e6)
        capex_vals.append(capex / 1e6)
        fcf_vals.append(fcf / 1e6)
        years.append(yr)

base_fcf = fcf_vals[0] * 1e6 if fcf_vals else 0.0

# Revenue & net income
rev_vals, ni_vals, inc_years = [], [], []
for stmt in inc_stmts[:4]:
    rev = raw(stmt, "totalRevenue")
    ni  = raw(stmt, "netIncome")
    yr  = stmt.get("endDate", {}).get("fmt", "")[:4]
    if rev is not None:
        rev_vals.append(rev / 1e6)
        ni_vals.append((ni or 0) / 1e6)
        inc_years.append(yr)

# Net debt
total_debt = raw(bal_stmts[0], "totalDebt") if bal_stmts else 0
cash       = raw(bal_stmts[0], "cash")       if bal_stmts else 0
net_debt   = (total_debt - cash) if (total_debt and cash) else 0.0

# ── Step 2 · Company Snapshot ─────────────────────────────────────────────────
st.header("Step 2 · Company Snapshot")
st.subheader(f"🏢 {company_name} ({ticker})")
st.caption(f"{sector} · {industry}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"${current_price:.2f}")
c2.metric("Market Cap",     fmt(market_cap))
c3.metric("Beta (live)",    f"{beta_live:.2f}")
c4.metric("Trailing P/E",   f"{pe_ratio:.1f}x" if pe_ratio else "N/A")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Forward P/E",    f"{fwd_pe:.1f}x"    if fwd_pe    else "N/A")
c6.metric("EV/EBITDA",      f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A")
c7.metric("Dividend Yield", f"{div_yield*100:.2f}%")
c8.metric("Shares Out",     fmt(shares_out).replace("$", ""))

if description:
    with st.expander("📋 Business Description"):
        st.write(description)

if not hist.empty:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"], mode="lines",
        line=dict(color="#4a6fa5", width=2), name="Close"
    ))
    fig_hist.update_layout(
        title=f"{ticker} – 2-Year Price History",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=280, margin=dict(t=40, b=20, l=20, r=20),
        template="plotly_white"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.divider()

# ── Step 3 · Financial Data ───────────────────────────────────────────────────
st.header("Step 3 · Financial Data (Auto-Retrieved)")

if fcf_vals:
    col_l, col_r = st.columns(2)
    with col_l:
        fcf_df = pd.DataFrame({
            "Year": years, "Operating CF": op_cf_vals,
            "CapEx": capex_vals, "Free Cash Flow": fcf_vals
        })
        fig_cf = go.Figure()
        fig_cf.add_bar(x=fcf_df["Year"], y=fcf_df["Operating CF"],
                       name="Operating CF", marker_color="#4a6fa5")
        fig_cf.add_bar(x=fcf_df["Year"], y=fcf_df["CapEx"],
                       name="CapEx", marker_color="#e07b54")
        fig_cf.add_scatter(x=fcf_df["Year"], y=fcf_df["Free Cash Flow"],
                           mode="lines+markers", name="FCF",
                           line=dict(color="#2ca02c", width=2, dash="dot"),
                           marker=dict(size=8))
        fig_cf.update_layout(title="Cash Flow Breakdown ($M)", barmode="group",
                              height=320, template="plotly_white",
                              margin=dict(t=40, b=20, l=20, r=20),
                              legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_cf, use_container_width=True)

    with col_r:
        if rev_vals:
            inc_df = pd.DataFrame({"Year": inc_years, "Revenue": rev_vals,
                                   "Net Income": ni_vals})
            fig_inc = go.Figure()
            fig_inc.add_bar(x=inc_df["Year"], y=inc_df["Revenue"],
                            name="Revenue", marker_color="#5b9bd5")
            fig_inc.add_bar(x=inc_df["Year"], y=inc_df["Net Income"],
                            name="Net Income", marker_color="#70ad47")
            fig_inc.update_layout(title="Revenue & Net Income ($M)", barmode="group",
                                  height=320, template="plotly_white",
                                  margin=dict(t=40, b=20, l=20, r=20),
                                  legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown(f"""
    <div class="metric-box">
    📌 <b>Base FCF (most recent):</b> {fmt(base_fcf)}&emsp;|&emsp;
    <b>Net Debt:</b> {fmt(net_debt)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-callout">
    <b>FCF = Operating Cash Flow − Capital Expenditures.</b>
    This is cash available after maintaining/growing the asset base —
    the foundation of the DCF model.
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("Cash flow data unavailable for this ticker. Enter Base FCF manually below.")

st.divider()

# ── Step 4 · WACC ─────────────────────────────────────────────────────────────
st.header("Step 4 · Cost of Capital (WACC)")
st.markdown("""
<div class="info-callout">
<b>WACC</b> is the blended required return of equity and debt holders.
It is used to discount future cash flows — a higher WACC means a lower valuation.
</div>
""", unsafe_allow_html=True)

with st.expander("📐 Formulas"):
    st.markdown("""
    <div class="formula-box">
    WACC = (E/V) × Re + (D/V) × Rd × (1 − Tax Rate)<br><br>
    Cost of Equity (CAPM): Re = Rf + β × (Rm − Rf)
    </div>
    """, unsafe_allow_html=True)

col_w1, col_w2 = st.columns(2)
with col_w1:
    st.markdown("#### CAPM Inputs")
    risk_free = st.slider("Risk-Free Rate Rf (%) — ~4.3% as of 2025",
                           0.5, 8.0, 4.3, 0.1,
                           help="10-yr US Treasury yield") / 100
    erp = st.slider("Equity Risk Premium ERP (%)",
                     2.0, 8.0, 5.5, 0.1,
                     help="Extra return investors demand over the risk-free rate. "
                          "Historical US average is ~5–6%.") / 100
    beta = st.number_input(f"Beta β  (live: {beta_live:.2f})",
                            0.1, 3.0, round(beta_live, 2), 0.05,
                            help="Measures stock volatility vs. the market.")
    re = compute_capm(risk_free, beta, erp)
    st.success(f"**Cost of Equity: {re*100:.2f}%**")
    st.caption(f"= {risk_free*100:.1f}% + {beta:.2f} × {erp*100:.1f}%")

with col_w2:
    st.markdown("#### Capital Structure")
    eq_pct   = st.slider("Equity Weight E/V (%)", 10, 100, 80, 5)
    debt_pct = 100 - eq_pct
    st.caption(f"⟹ Debt Weight D/V = {debt_pct}%")
    rd = st.slider("Pre-Tax Cost of Debt Rd (%)", 1.0, 12.0, 4.0, 0.25,
                   help="Yield on the company's outstanding debt.") / 100
    tax = st.slider("Effective Tax Rate (%)", 5, 40, 21, 1,
                    help="US statutory rate is 21%; effective rates vary.") / 100
    wacc = compute_wacc(eq_pct/100, debt_pct/100, re, rd, tax)
    st.success(f"**WACC: {wacc*100:.2f}%**")

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(wacc * 100, 2),
    title={"text": "WACC (%)"},
    gauge={
        "axis": {"range": [0, 20]},
        "bar": {"color": "#4a6fa5"},
        "steps": [
            {"range": [0,  6], "color": "#d4edda"},
            {"range": [6, 12], "color": "#fff3cd"},
            {"range": [12,20], "color": "#f8d7da"},
        ],
    },
    number={"suffix": "%", "font": {"size": 36}},
))
fig_gauge.update_layout(height=230, margin=dict(t=20, b=10))
st.plotly_chart(fig_gauge, use_container_width=True)
st.divider()

# ── Step 5 · Growth Assumptions ──────────────────────────────────────────────
st.header("Step 5 · FCF Growth Assumptions")
st.markdown("""
<div class="info-callout">
<b>Two-stage model:</b> Stage 1 = explicit year-by-year FCF growth (you control).
Stage 2 = terminal value via Gordon Growth Model (perpetuity beyond Stage 1).
</div>
""", unsafe_allow_html=True)

col_g1, col_g2 = st.columns([2, 1])
with col_g1:
    st.markdown("#### Stage 1 — Explicit Forecast Period")
    forecast_years = st.slider("Number of forecast years", 3, 10, 5)
    use_custom = st.toggle("Set custom rate per year")
    growth_rates = []
    if use_custom:
        cols = st.columns(forecast_years)
        for i, c in enumerate(cols):
            g = c.number_input(f"Yr {i+1} (%)", -30.0, 50.0,
                                15.0 if i < 3 else 10.0, 0.5, key=f"g{i}")
            growth_rates.append(g / 100)
    else:
        g_hi = st.slider("Near-term growth Yrs 1–3 (%)", -10.0, 40.0, 15.0, 0.5) / 100
        g_md = st.slider("Mid-term growth Yrs 4+ (%)",    -10.0, 30.0, 10.0, 0.5) / 100
        growth_rates = [g_hi] * min(3, forecast_years) + \
                       [g_md] * max(0, forecast_years - 3)

with col_g2:
    st.markdown("#### Stage 2 — Terminal Value")
    terminal_growth = st.slider("Perpetuity Growth Rate (%)", 0.0, 5.0, 2.5, 0.1,
                                 help="Must be below WACC. ~Nominal GDP growth (2–3%).") / 100
    if terminal_growth >= wacc:
        st.error("⚠️ Terminal growth must be < WACC!")
    st.markdown("""
    <div class="info-callout">
    TV = FCF_n × (1+g) / (WACC − g)
    </div>
    """, unsafe_allow_html=True)

st.markdown("#### Base FCF Override")
base_fcf_input = st.number_input(
    f"Base FCF ($M) — auto-pulled: {base_fcf/1e6:.1f}M",
    min_value=-50000.0, max_value=500000.0,
    value=round(base_fcf / 1e6, 1), step=50.0,
    help="Override if you want to normalize for one-time items."
) * 1e6

st.divider()

# ── Step 6 · DCF Results ──────────────────────────────────────────────────────
st.header("Step 6 · DCF Output & Valuation")

if terminal_growth >= wacc:
    st.error("Fix the terminal growth rate before viewing results.")
    st.stop()

result = run_dcf(base_fcf_input, growth_rates, terminal_growth,
                 wacc, shares_out, net_debt)

dcf_df = pd.DataFrame(result["rows"])
st.markdown("### 📋 Year-by-Year Projections")
st.dataframe(dcf_df.style.format({
    "FCF ($M)": "{:,.1f}", "PV of FCF ($M)": "{:,.1f}",
    "Discount Factor": "{:.4f}"}),
    use_container_width=True, hide_index=True)

pv1  = result["pv_stage1"]
tv   = result["terminal_value"]
tv_p = result["tv_pv"]
ev   = result["enterprise_value"]
eqv  = result["equity_value"]
iv   = result["intrinsic_per_share"]
updown = (iv - current_price) / current_price * 100

c1, c2, c3 = st.columns(3)
c1.metric("PV Stage 1 FCFs",       fmt(pv1))
c2.metric("Terminal Value (gross)", fmt(tv))
c3.metric("PV of Terminal Value",   fmt(tv_p))

c4, c5, c6 = st.columns(3)
c4.metric("Enterprise Value", fmt(ev))
c5.metric("(−) Net Debt",     fmt(net_debt))
c6.metric("Equity Value",     fmt(eqv))

st.markdown("---")
st.subheader("🎯 Intrinsic Value Per Share")

col_v, col_wf = st.columns(2)
with col_v:
    st.metric("Intrinsic Value (DCF)", f"${iv:.2f}",
              delta=f"{updown:+.1f}% vs market ${current_price:.2f}")
    mos = (iv - current_price) / iv * 100 if iv else 0
    if iv > current_price * 1.15:
        st.markdown(f"""<div class="verdict-undervalued">
        ✅ <b>POTENTIALLY UNDERVALUED</b><br>
        Trading at a <b>{abs(updown):.1f}% discount</b>.
        Margin of safety: <b>{mos:.1f}%</b>.
        </div>""", unsafe_allow_html=True)
    elif iv < current_price * 0.85:
        st.markdown(f"""<div class="verdict-overvalued">
        ❌ <b>POTENTIALLY OVERVALUED</b><br>
        Trading at a <b>{abs(updown):.1f}% premium</b> to intrinsic value.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="verdict-fair">
        ⚖️ <b>APPROXIMATELY FAIRLY VALUED</b><br>
        Within 15% of intrinsic value.
        </div>""", unsafe_allow_html=True)

with col_wf:
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        x=["PV Stage 1", "PV Terminal", "Enterprise Value", "Less Net Debt", "Equity Value"],
        y=[pv1/1e9, tv_p/1e9, 0, -net_debt/1e9, 0],
        measure=["relative","relative","total","relative","total"],
        text=[f"${v:.1f}B" for v in [pv1/1e9, tv_p/1e9, ev/1e9, -net_debt/1e9, eqv/1e9]],
        textposition="outside",
        connector={"line": {"color": "rgb(63,63,63)"}},
        decreasing={"marker": {"color": "#e07b54"}},
        increasing={"marker": {"color": "#4a6fa5"}},
        totals={"marker": {"color": "#2ca02c"}},
    ))
    fig_wf.update_layout(title="Valuation Build-Up ($B)", height=350,
                          template="plotly_white",
                          margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_wf, use_container_width=True)

st.divider()

# ── Step 7 · Sensitivity Analysis ─────────────────────────────────────────────
st.header("Step 7 · Sensitivity Analysis")
st.markdown("""
<div class="info-callout">
How does intrinsic value change as WACC and terminal growth vary?
Green = above market price; Red = below market price.
</div>
""", unsafe_allow_html=True)

wacc_range = np.arange(wacc - 0.03, wacc + 0.035, 0.005)
tg_range   = np.arange(max(0.005, terminal_growth - 0.015),
                        min(wacc - 0.01, terminal_growth + 0.02), 0.005)

matrix = []
for tg in tg_range:
    row = []
    for w in wacc_range:
        if tg >= w:
            row.append(np.nan)
        else:
            r = run_dcf(base_fcf_input, growth_rates, tg, w, shares_out, net_debt)
            row.append(round(r["intrinsic_per_share"], 2))
    matrix.append(row)

sens_df = pd.DataFrame(
    matrix,
    index=[f"{tg*100:.1f}%" for tg in tg_range],
    columns=[f"{w*100:.1f}%" for w in wacc_range],
)

def color_cell(val):
    if pd.isna(val):
        return "background-color:#ccc"
    if val > current_price * 1.15:
        return "background-color:#d4edda;color:black"
    elif val < current_price * 0.85:
        return "background-color:#f8d7da;color:black"
    return "background-color:#fff3cd;color:black"

st.markdown("**IV/share · columns = WACC · rows = Terminal Growth**")
st.dataframe(sens_df.style.applymap(color_cell).format("${:.2f}", na_rep="—"),
             use_container_width=True)
st.divider()

# ── Step 8 · Scenarios ────────────────────────────────────────────────────────
st.header("Step 8 · Bear / Base / Bull Scenarios")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.markdown("🐻 **Bear**")
    b_g  = st.slider("Bear growth (%)", -10, 30, max(0, int(growth_rates[0]*100-8)), key="bg") / 100
    b_w  = st.slider("Bear WACC (%)",    5, 20, min(20, int(wacc*100+2)), key="bw") / 100
    b_tg = st.slider("Bear terminal (%)",0, 4, 1, key="btg") / 100
with sc2:
    st.markdown("⚖️ **Base (your inputs)**")
    st.metric("Near-term growth", f"{growth_rates[0]*100:.1f}%")
    st.metric("WACC",             f"{wacc*100:.2f}%")
    st.metric("Terminal growth",  f"{terminal_growth*100:.1f}%")
with sc3:
    st.markdown("🐂 **Bull**")
    u_g  = st.slider("Bull growth (%)", 0, 60, min(50, int(growth_rates[0]*100+8)), key="ug") / 100
    u_w  = st.slider("Bull WACC (%)",   4, 18, max(4,  int(wacc*100-2)), key="uw") / 100
    u_tg = st.slider("Bull terminal (%)",1, 5, 3, key="utg") / 100

def sc_rates(g, n):
    return [g]*min(3,n) + [max(0,g*0.7)]*max(0,n-3)

bear_r = run_dcf(base_fcf_input, sc_rates(b_g, forecast_years), b_tg, b_w, shares_out, net_debt)
bull_r = run_dcf(base_fcf_input, sc_rates(u_g, forecast_years), u_tg, u_w, shares_out, net_debt)

scen_vals = [bear_r["intrinsic_per_share"], iv, bull_r["intrinsic_per_share"]]
scen_lbls = ["🐻 Bear", "⚖️ Base", "🐂 Bull"]
colors    = ["#e07b54", "#4a6fa5", "#2ca02c"]

fig_sc = go.Figure()
for lbl, val, col in zip(scen_lbls, scen_vals, colors):
    fig_sc.add_bar(x=[lbl], y=[val], name=lbl, marker_color=col,
                   text=[f"${val:.2f}"], textposition="outside")
fig_sc.add_hline(y=current_price, line_dash="dash",
                  annotation_text=f"Market ${current_price:.2f}",
                  annotation_position="top right")
fig_sc.update_layout(title="Intrinsic Value — Scenarios", yaxis_title="$/share",
                      height=350, template="plotly_white", showlegend=False,
                      margin=dict(t=50, b=20, l=20, r=20))
st.plotly_chart(fig_sc, use_container_width=True)

scen_df = pd.DataFrame({
    "Scenario": scen_lbls,
    "Intrinsic Value": [f"${v:.2f}" for v in scen_vals],
    "vs. Market": [f"{(v-current_price)/current_price*100:+.1f}%" for v in scen_vals],
})
st.dataframe(scen_df.set_index("Scenario"), use_container_width=True)
st.divider()

# ── Step 9 · Relative Multiples ───────────────────────────────────────────────
st.header("Step 9 · Relative Valuation Benchmarks")
st.markdown("""
<div class="info-callout">
Multiples provide a quick sanity check alongside the DCF. They don't replace
intrinsic value analysis but help contextualize your findings.
</div>
""", unsafe_allow_html=True)

pb = raw(key_mod, "priceToBook")
st.dataframe(pd.DataFrame({
    "Metric":        ["Trailing P/E", "Forward P/E", "EV/EBITDA", "Price/Book"],
    ticker:          [
        f"{pe_ratio:.1f}x" if pe_ratio  else "N/A",
        f"{fwd_pe:.1f}x"   if fwd_pe    else "N/A",
        f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A",
        f"{pb:.1f}x"       if pb        else "N/A",
    ],
    "S&P 500 Avg":   ["~22x", "~20x", "~14x", "~4x"],
    "Interpretation": [
        "<15x cheap · 15–25x fair · >25x premium",
        "Lower = more attractively priced on fwd earnings",
        "<10x cheap · 10–20x fair · >20x premium",
        "<1x deep value · >5x growth premium",
    ],
}).set_index("Metric"), use_container_width=True)
st.divider()

# ── Step 10 · Full Walkthrough ────────────────────────────────────────────────
st.header("Step 10 · Full Calculation Walkthrough")
with st.expander("📖 See every formula with your actual numbers"):
    st.markdown(f"""
#### 1. Base Free Cash Flow
- Operating CF − CapEx = **{fmt(base_fcf_input)}**

#### 2. Cost of Equity (CAPM)
- Re = {risk_free*100:.1f}% + {beta:.2f} × {erp*100:.1f}% = **{re*100:.2f}%**

#### 3. WACC
- WACC = ({eq_pct}% × {re*100:.2f}%) + ({debt_pct}% × {rd*100:.2f}% × (1−{tax*100:.0f}%)) = **{wacc*100:.2f}%**

#### 4. Stage 1 — Year-by-Year FCF
""")
    for r in result["rows"]:
        st.markdown(
            f"- **Year {r['Year']}** · Growth {r['Growth Rate']} → "
            f"FCF ${r['FCF ($M)']:,.1f}M · "
            f"Discount factor {r['Discount Factor']:.4f} → "
            f"PV **${r['PV of FCF ($M)']:,.1f}M**"
        )
    st.markdown(f"""
#### 5. Terminal Value
- Last FCF × (1+{terminal_growth*100:.1f}%) / ({wacc*100:.2f}%−{terminal_growth*100:.1f}%) = **{fmt(tv)}**
- PV of Terminal Value = **{fmt(tv_p)}**

#### 6. Enterprise Value
- {fmt(pv1)} (Stage 1) + {fmt(tv_p)} (TV) = **{fmt(ev)}**

#### 7. Equity Value & Per-Share Value
- {fmt(ev)} − {fmt(net_debt)} (net debt) = **{fmt(eqv)}**
- {fmt(eqv)} ÷ {fmt(shares_out).replace('$','')} shares = **${iv:.2f}/share**

#### 8. Verdict
- Market Price: **${current_price:.2f}** · Intrinsic Value: **${iv:.2f}** · Δ **{updown:+.1f}%**
""")

st.divider()
st.caption("⚠️ For educational purposes only. Not investment advice.")
