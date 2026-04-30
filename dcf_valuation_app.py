import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Equity Valuation | DCF Calculator",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .formula-box {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 6px; padding: 14px;
        font-family: monospace; font-size: 0.9em; margin: 8px 0;
    }
    .verdict-undervalued {
        background: #d4edda; border-left: 5px solid #28a745;
        padding: 12px 18px; border-radius: 6px; font-size: 1.1em;
    }
    .verdict-overvalued {
        background: #f8d7da; border-left: 5px solid #dc3545;
        padding: 12px 18px; border-radius: 6px; font-size: 1.1em;
    }
    .verdict-fair {
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 12px 18px; border-radius: 6px; font-size: 1.1em;
    }
    .info-callout {
        background: #e8f4f8; border-left: 4px solid #17a2b8;
        padding: 10px 14px; border-radius: 4px; font-size: 0.9em;
    }
    .metric-box {
        background: #f0f4ff; border-left: 4px solid #4a6fa5;
        padding: 12px 16px; border-radius: 6px; margin: 6px 0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data(ticker: str):
    tk   = yf.Ticker(ticker)
    info = tk.info
    cf   = tk.cashflow
    inc  = tk.income_stmt
    bal  = tk.balance_sheet
    hist = tk.history(period="2y")
    return info, cf, inc, bal, hist


def safe_get(df, row_label, col_index=0, default=None):
    try:
        val = df.loc[row_label].iloc[col_index]
        return float(val) if not pd.isna(val) else default
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


# ═══════════════════════════════════════════════════════════════
# DCF ENGINE
# ═══════════════════════════════════════════════════════════════

def capm(rf, beta, erp):
    return rf + beta * erp


def wacc_calc(eq_w, d_w, re, rd, tax):
    return eq_w * re + d_w * rd * (1 - tax)


def run_dcf(base_fcf, growth_rates, tg, wacc, shares, net_debt):
    rows, pv_sum, fcf = [], 0.0, base_fcf
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
    tv    = fcf * (1 + tg) / (wacc - tg)
    tv_pv = tv / (1 + wacc) ** len(growth_rates)
    pv_sum += tv_pv
    ev  = pv_sum
    eqv = ev - net_debt
    iv  = eqv / shares if shares else 0
    return dict(rows=rows, terminal_value=tv, tv_pv=tv_pv,
                enterprise_value=ev, equity_value=eqv,
                intrinsic_per_share=iv, pv_stage1=pv_sum - tv_pv)


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

st.title("📊 Equity Intrinsic Value Calculator")
st.markdown("**Two-Stage DCF Model** · FINA 4011/5011 Project 2")
st.markdown("""<div class="info-callout">
Enter a ticker below. The app fetches live financial data from Yahoo Finance,
then walks you through every step of a two-stage DCF valuation.
</div>""", unsafe_allow_html=True)
st.divider()

# ── Step 1 · Ticker ───────────────────────────────────────────
st.header("Step 1 · Select a Stock")
col_t, col_b = st.columns([2, 1])
with col_t:
    ticker_input = st.text_input(
        "Stock ticker (e.g. AAPL, MSFT, TSLA)", value="AAPL"
    ).upper().strip()
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    load_btn = st.button("🔄 Load Data", use_container_width=True)

if "ticker" not in st.session_state:
    st.session_state.ticker = ticker_input
if load_btn:
    st.session_state.ticker = ticker_input
ticker = st.session_state.ticker

# ── Fetch ─────────────────────────────────────────────────────
with st.spinner(f"Fetching live data for **{ticker}**…"):
    try:
        info, cf, inc, bal, hist = fetch_stock_data(ticker)
    except Exception as e:
        st.error(f"Could not load data for **{ticker}**. Check the ticker.\n\n{e}")
        st.stop()

if not info or not info.get("regularMarketPrice") and not info.get("currentPrice"):
    st.error("Ticker not found or no price data available.")
    st.stop()

# Parse info
company_name  = info.get("longName", ticker)
current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
market_cap    = info.get("marketCap", 0)
beta_live     = info.get("beta", 1.0) or 1.0
shares_out    = info.get("sharesOutstanding", 1)
pe_ratio      = info.get("trailingPE")
fwd_pe        = info.get("forwardPE")
ev_ebitda     = info.get("enterpriseToEbitda")
div_yield     = info.get("dividendYield", 0) or 0
pb            = info.get("priceToBook")
sector        = info.get("sector", "N/A")
industry      = info.get("industry", "N/A")
description   = info.get("longBusinessSummary", "")

# Cash flows
fcf_vals, years, op_cf_vals, capex_vals = [], [], [], []
try:
    for i in range(min(4, cf.shape[1])):
        ocf   = safe_get(cf, "Operating Cash Flow", i)
        capex = safe_get(cf, "Capital Expenditure", i)
        if ocf is not None and capex is not None:
            op_cf_vals.append(ocf / 1e6)
            capex_vals.append(capex / 1e6)
            fcf_vals.append((ocf + capex) / 1e6)
            years.append(cf.columns[i].year)
except Exception:
    pass

base_fcf = fcf_vals[0] * 1e6 if fcf_vals else 0.0

rev_vals, ni_vals, inc_years = [], [], []
try:
    for i in range(min(4, inc.shape[1])):
        rev = safe_get(inc, "Total Revenue", i)
        ni  = safe_get(inc, "Net Income", i)
        if rev:
            rev_vals.append(rev / 1e6)
            ni_vals.append((ni or 0) / 1e6)
            inc_years.append(inc.columns[i].year)
except Exception:
    pass

try:
    total_debt = safe_get(bal, "Total Debt", 0, 0)
    cash_      = safe_get(bal, "Cash And Cash Equivalents", 0, 0)
    net_debt   = (total_debt - cash_) if (total_debt and cash_) else 0.0
except Exception:
    net_debt = 0.0

# ── Step 2 · Snapshot ─────────────────────────────────────────
st.header("Step 2 · Company Snapshot")
st.subheader(f"🏢 {company_name} ({ticker})")
st.caption(f"{sector} · {industry}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"${current_price:.2f}")
c2.metric("Market Cap",     fmt(market_cap))
c3.metric("Beta",           f"{beta_live:.2f}")
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
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines",
                               line=dict(color="#4a6fa5", width=2)))
    fig_h.update_layout(title=f"{ticker} – 2-Year Price History",
                         xaxis_title="Date", yaxis_title="USD",
                         height=260, template="plotly_white",
                         margin=dict(t=40, b=10, l=10, r=10))
    st.plotly_chart(fig_h, use_container_width=True)

st.divider()

# ── Step 3 · Financials ────────────────────────────────────────
st.header("Step 3 · Financial Data (Auto-Retrieved)")

if fcf_vals:
    cl, cr = st.columns(2)
    with cl:
        fig_cf = go.Figure()
        fig_cf.add_bar(x=years, y=op_cf_vals, name="Operating CF", marker_color="#4a6fa5")
        fig_cf.add_bar(x=years, y=capex_vals, name="CapEx",        marker_color="#e07b54")
        fig_cf.add_scatter(x=years, y=fcf_vals, mode="lines+markers", name="FCF",
                           line=dict(color="#2ca02c", width=2, dash="dot"),
                           marker=dict(size=8))
        fig_cf.update_layout(title="Cash Flow ($M)", barmode="group", height=300,
                              template="plotly_white",
                              margin=dict(t=40, b=10, l=10, r=10),
                              legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_cf, use_container_width=True)
    with cr:
        if rev_vals:
            fig_inc = go.Figure()
            fig_inc.add_bar(x=inc_years, y=rev_vals, name="Revenue",   marker_color="#5b9bd5")
            fig_inc.add_bar(x=inc_years, y=ni_vals,  name="Net Income", marker_color="#70ad47")
            fig_inc.update_layout(title="Income ($M)", barmode="group", height=300,
                                  template="plotly_white",
                                  margin=dict(t=40, b=10, l=10, r=10),
                                  legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig_inc, use_container_width=True)

    st.markdown(f"""<div class="metric-box">
    📌 <b>Base FCF (most recent year):</b> {fmt(base_fcf)}&emsp;|&emsp;
    <b>Net Debt:</b> {fmt(net_debt)}
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class="info-callout">
    <b>FCF = Operating Cash Flow − Capital Expenditures.</b>
    This is cash available after maintaining/growing assets — the foundation of DCF.
    </div>""", unsafe_allow_html=True)
else:
    st.warning("Cash flow data unavailable. Enter Base FCF manually in Step 5.")

st.divider()

# ── Step 4 · WACC ─────────────────────────────────────────────
st.header("Step 4 · Cost of Capital (WACC)")
st.markdown("""<div class="info-callout">
WACC is the blended required return of all capital providers. It is the discount
rate applied to future cash flows — a higher WACC lowers the valuation.
</div>""", unsafe_allow_html=True)

with st.expander("📐 Formulas"):
    st.markdown("""<div class="formula-box">
    WACC = (E/V) × Re + (D/V) × Rd × (1 − Tax Rate)<br><br>
    Cost of Equity (CAPM): Re = Rf + β × ERP
    </div>""", unsafe_allow_html=True)

cw1, cw2 = st.columns(2)
with cw1:
    st.markdown("#### CAPM")
    rf  = st.slider("Risk-Free Rate Rf (%) — 10-yr Treasury, ~4.3% in 2025",
                    0.5, 8.0, 4.3, 0.1) / 100
    erp = st.slider("Equity Risk Premium ERP (%) — historical US avg ~5.5%",
                    2.0, 8.0, 5.5, 0.1) / 100
    beta = st.number_input(f"Beta β (live: {beta_live:.2f})",
                            0.1, 3.0, round(beta_live, 2), 0.05)
    re = capm(rf, beta, erp)
    st.success(f"**Cost of Equity: {re*100:.2f}%**")
    st.caption(f"= {rf*100:.1f}% + {beta:.2f} × {erp*100:.1f}%")

with cw2:
    st.markdown("#### Capital Structure")
    eq_pct   = st.slider("Equity Weight E/V (%)", 10, 100, 80, 5)
    debt_pct = 100 - eq_pct
    st.caption(f"⟹ Debt Weight D/V = {debt_pct}%")
    rd  = st.slider("Pre-Tax Cost of Debt Rd (%)", 1.0, 12.0, 4.0, 0.25) / 100
    tax = st.slider("Effective Tax Rate (%)", 5, 40, 21, 1) / 100
    wacc = wacc_calc(eq_pct/100, debt_pct/100, re, rd, tax)
    st.success(f"**WACC: {wacc*100:.2f}%**")

fig_g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(wacc * 100, 2),
    title={"text": "WACC (%)"},
    gauge={"axis": {"range": [0, 20]}, "bar": {"color": "#4a6fa5"},
           "steps": [{"range": [0,  6], "color": "#d4edda"},
                     {"range": [6, 12], "color": "#fff3cd"},
                     {"range": [12,20], "color": "#f8d7da"}]},
    number={"suffix": "%", "font": {"size": 36}},
))
fig_g.update_layout(height=220, margin=dict(t=10, b=10))
st.plotly_chart(fig_g, use_container_width=True)
st.divider()

# ── Step 5 · Growth ────────────────────────────────────────────
st.header("Step 5 · FCF Growth Assumptions")
st.markdown("""<div class="info-callout">
<b>Two-stage:</b> Stage 1 = explicit year-by-year growth you set.
Stage 2 = terminal value via Gordon Growth Model.
</div>""", unsafe_allow_html=True)

cg1, cg2 = st.columns([2, 1])
with cg1:
    st.markdown("#### Stage 1 — Explicit Period")
    n_years    = st.slider("Forecast years", 3, 10, 5)
    use_custom = st.toggle("Custom rate per year")
    growth_rates = []
    if use_custom:
        cols = st.columns(n_years)
        for i, c in enumerate(cols):
            g = c.number_input(f"Yr {i+1} (%)", -30.0, 50.0,
                                15.0 if i < 3 else 10.0, 0.5, key=f"g{i}")
            growth_rates.append(g / 100)
    else:
        g_hi = st.slider("Near-term growth Yrs 1–3 (%)", -10.0, 40.0, 15.0, 0.5) / 100
        g_md = st.slider("Mid-term growth Yrs 4+ (%)",    -10.0, 30.0, 10.0, 0.5) / 100
        growth_rates = [g_hi]*min(3, n_years) + [g_md]*max(0, n_years - 3)

with cg2:
    st.markdown("#### Stage 2 — Terminal")
    tg = st.slider("Perpetuity Growth Rate (%)", 0.0, 5.0, 2.5, 0.1,
                   help="Must be below WACC. Typically ~nominal GDP growth (2–3%).") / 100
    if tg >= wacc:
        st.error("⚠️ Terminal growth must be < WACC!")
    st.markdown("""<div class="info-callout">
    TV = FCF_n × (1+g) / (WACC − g)
    </div>""", unsafe_allow_html=True)

st.markdown("#### Base FCF")
base_fcf_input = st.number_input(
    f"Base FCF ($M) — auto-pulled: {base_fcf/1e6:.1f}M",
    min_value=-50000.0, max_value=500000.0,
    value=round(base_fcf / 1e6, 1), step=50.0,
    help="Starting point for projections. Override to normalize one-time items."
) * 1e6

st.divider()

# ── Step 6 · Results ───────────────────────────────────────────
st.header("Step 6 · DCF Results")

if tg >= wacc:
    st.error("Fix terminal growth rate before continuing.")
    st.stop()

res = run_dcf(base_fcf_input, growth_rates, tg, wacc, shares_out, net_debt)

st.markdown("### 📋 Year-by-Year Projections")
dcf_df = pd.DataFrame(res["rows"])
st.dataframe(dcf_df.style.format({
    "FCF ($M)": "{:,.1f}", "PV of FCF ($M)": "{:,.1f}",
    "Discount Factor": "{:.4f}"}),
    use_container_width=True, hide_index=True)

pv1, tv, tv_p = res["pv_stage1"], res["terminal_value"], res["tv_pv"]
ev, eqv, iv   = res["enterprise_value"], res["equity_value"], res["intrinsic_per_share"]
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

cv, cwf = st.columns(2)
with cv:
    st.metric("Intrinsic Value (DCF)", f"${iv:.2f}",
              delta=f"{updown:+.1f}% vs market ${current_price:.2f}")
    mos = (iv - current_price) / iv * 100 if iv else 0
    if iv > current_price * 1.15:
        st.markdown(f"""<div class="verdict-undervalued">
        ✅ <b>POTENTIALLY UNDERVALUED</b><br>
        {abs(updown):.1f}% discount · Margin of safety: {mos:.1f}%
        </div>""", unsafe_allow_html=True)
    elif iv < current_price * 0.85:
        st.markdown(f"""<div class="verdict-overvalued">
        ❌ <b>POTENTIALLY OVERVALUED</b><br>
        Trading at a {abs(updown):.1f}% premium to intrinsic value.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="verdict-fair">
        ⚖️ <b>APPROXIMATELY FAIRLY VALUED</b><br>
        Within 15% of intrinsic value.
        </div>""", unsafe_allow_html=True)

with cwf:
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        x=["PV Stage 1", "PV Terminal", "Enterprise Value",
           "Less Net Debt", "Equity Value"],
        y=[pv1/1e9, tv_p/1e9, 0, -net_debt/1e9, 0],
        measure=["relative","relative","total","relative","total"],
        text=[f"${v:.1f}B" for v in
              [pv1/1e9, tv_p/1e9, ev/1e9, -net_debt/1e9, eqv/1e9]],
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

# ── Step 7 · Sensitivity ───────────────────────────────────────
st.header("Step 7 · Sensitivity Analysis")
st.markdown("""<div class="info-callout">
Intrinsic value per share across a range of WACC and terminal growth inputs.
🟢 Above market price · 🔴 Below market price · 🟡 Within 15%
</div>""", unsafe_allow_html=True)

wacc_range = np.arange(wacc - 0.03, wacc + 0.035, 0.005)
tg_range   = np.arange(max(0.005, tg - 0.015),
                        min(wacc - 0.01, tg + 0.02), 0.005)

matrix = []
for tg_ in tg_range:
    row = []
    for w in wacc_range:
        if tg_ >= w:
            row.append(np.nan)
        else:
            r = run_dcf(base_fcf_input, growth_rates, tg_, w, shares_out, net_debt)
            row.append(round(r["intrinsic_per_share"], 2))
    matrix.append(row)

sens_df = pd.DataFrame(
    matrix,
    index=[f"{v*100:.1f}%" for v in tg_range],
    columns=[f"{v*100:.1f}%" for v in wacc_range],
)

def color_cell(val):
    if pd.isna(val):
        return "background-color:#ccc"
    if val > current_price * 1.15:
        return "background-color:#d4edda;color:black"
    if val < current_price * 0.85:
        return "background-color:#f8d7da;color:black"
    return "background-color:#fff3cd;color:black"

st.markdown("**Columns = WACC · Rows = Terminal Growth**")
st.dataframe(sens_df.style.applymap(color_cell).format("${:.2f}", na_rep="—"),
             use_container_width=True)
st.divider()

# ── Step 8 · Scenarios ─────────────────────────────────────────
st.header("Step 8 · Bear / Base / Bull Scenarios")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    st.markdown("🐻 **Bear**")
    b_g  = st.slider("Growth (%)", -10, 30, max(0, int(growth_rates[0]*100-8)), key="bg") / 100
    b_w  = st.slider("WACC (%)",    5, 20, min(20, int(wacc*100+2)), key="bw") / 100
    b_tg = st.slider("Terminal (%)", 0, 4, 1, key="btg") / 100
with sc2:
    st.markdown("⚖️ **Base (your inputs)**")
    st.metric("Near-term growth", f"{growth_rates[0]*100:.1f}%")
    st.metric("WACC",             f"{wacc*100:.2f}%")
    st.metric("Terminal growth",  f"{tg*100:.1f}%")
with sc3:
    st.markdown("🐂 **Bull**")
    u_g  = st.slider("Growth (%)", 0, 60, min(50, int(growth_rates[0]*100+8)), key="ug") / 100
    u_w  = st.slider("WACC (%)",   4, 18, max(4,  int(wacc*100-2)), key="uw") / 100
    u_tg = st.slider("Terminal (%)", 1, 5, 3, key="utg") / 100

def sc_r(g, n):
    return [g]*min(3,n) + [max(0,g*0.7)]*max(0,n-3)

bear_r = run_dcf(base_fcf_input, sc_r(b_g, n_years), b_tg, b_w, shares_out, net_debt)
bull_r = run_dcf(base_fcf_input, sc_r(u_g, n_years), u_tg, u_w, shares_out, net_debt)

vals  = [bear_r["intrinsic_per_share"], iv, bull_r["intrinsic_per_share"]]
lbls  = ["🐻 Bear", "⚖️ Base", "🐂 Bull"]
cols_ = ["#e07b54", "#4a6fa5", "#2ca02c"]

fig_sc = go.Figure()
for l, v, c in zip(lbls, vals, cols_):
    fig_sc.add_bar(x=[l], y=[v], marker_color=c,
                   text=[f"${v:.2f}"], textposition="outside")
fig_sc.add_hline(y=current_price, line_dash="dash",
                  annotation_text=f"Market ${current_price:.2f}",
                  annotation_position="top right")
fig_sc.update_layout(title="Intrinsic Value — Scenarios", yaxis_title="$/share",
                      height=340, template="plotly_white", showlegend=False,
                      margin=dict(t=50, b=20, l=20, r=20))
st.plotly_chart(fig_sc, use_container_width=True)

st.dataframe(pd.DataFrame({
    "Scenario": lbls,
    "Intrinsic Value": [f"${v:.2f}" for v in vals],
    "vs. Market": [f"{(v-current_price)/current_price*100:+.1f}%" for v in vals],
}).set_index("Scenario"), use_container_width=True)
st.divider()

# ── Step 9 · Multiples ─────────────────────────────────────────
st.header("Step 9 · Relative Valuation Benchmarks")
st.dataframe(pd.DataFrame({
    "Metric": ["Trailing P/E","Forward P/E","EV/EBITDA","Price/Book"],
    ticker: [
        f"{pe_ratio:.1f}x"  if pe_ratio  else "N/A",
        f"{fwd_pe:.1f}x"    if fwd_pe    else "N/A",
        f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A",
        f"{pb:.1f}x"        if pb        else "N/A",
    ],
    "S&P 500 Avg": ["~22x","~20x","~14x","~4x"],
    "Interpretation": [
        "<15x cheap · 15–25x fair · >25x premium",
        "Lower = more attractively priced",
        "<10x cheap · 10–20x fair · >20x premium",
        "<1x deep value · >5x growth premium",
    ],
}).set_index("Metric"), use_container_width=True)
st.divider()

# ── Step 10 · Walkthrough ──────────────────────────────────────
st.header("Step 10 · Full Calculation Walkthrough")
with st.expander("📖 Every formula with your actual numbers"):
    st.markdown(f"""
#### 1. Base FCF
Operating CF − CapEx = **{fmt(base_fcf_input)}**

#### 2. Cost of Equity (CAPM)
Re = {rf*100:.1f}% + {beta:.2f} × {erp*100:.1f}% = **{re*100:.2f}%**

#### 3. WACC
({eq_pct}% × {re*100:.2f}%) + ({debt_pct}% × {rd*100:.2f}% × (1−{tax*100:.0f}%)) = **{wacc*100:.2f}%**

#### 4. Stage 1 FCF Projections
""")
    for r in res["rows"]:
        st.markdown(
            f"- **Year {r['Year']}** · {r['Growth Rate']} growth → "
            f"FCF ${r['FCF ($M)']:,.1f}M · "
            f"DF {r['Discount Factor']:.4f} → "
            f"PV **${r['PV of FCF ($M)']:,.1f}M**"
        )
    st.markdown(f"""
#### 5. Terminal Value
FCF_n × (1+{tg*100:.1f}%) / ({wacc*100:.2f}%−{tg*100:.1f}%) = **{fmt(tv)}**
PV of Terminal Value = **{fmt(tv_p)}**

#### 6. Enterprise Value
{fmt(pv1)} + {fmt(tv_p)} = **{fmt(ev)}**

#### 7. Equity Value → Per Share
{fmt(ev)} − {fmt(net_debt)} = **{fmt(eqv)}**
{fmt(eqv)} ÷ {fmt(shares_out).replace('$','')} shares = **${iv:.2f}**

#### 8. Verdict
Market Price **${current_price:.2f}** · Intrinsic Value **${iv:.2f}** · Δ **{updown:+.1f}%**
""")

st.divider()
st.caption("⚠️ For educational purposes only. Not investment advice.")
