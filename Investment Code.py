import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
#from anthropic import Anthropic

st.set_page_config(page_title="DCF Valuation Model", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
    .verdict-box { padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
    .undervalued { background: #e8f5e9; border-left: 4px solid #2e7d32; }
    .overvalued  { background: #ffebee; border-left: 4px solid #c62828; }
    .fair        { background: #fff8e1; border-left: 4px solid #f57f17; }
    h1 { font-size: 1.8rem !important; }
    .step-box { background: #f0f4f8; border-left: 3px solid #1976d2; padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────
st.title("📈 DCF Intrinsic Value Calculator")
st.caption("Discounted Cash Flow model powered by live AI data retrieval")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Anthropic API Key", type="password",
                            help="Get yours at console.anthropic.com")
    st.markdown("---")
    st.header("📌 What is DCF?")
    st.info(
        "**Discounted Cash Flow (DCF)** estimates what a company is worth today "
        "by projecting its future earnings and discounting them back to present value "
        "using a required rate of return (WACC).\n\n"
        "**Intrinsic value > market price** → potentially undervalued\n\n"
        "**Intrinsic value < market price** → potentially overvalued"
    )
    st.markdown("---")
    st.caption("⚠️ For educational purposes only. Not financial advice.")

# ── Step 1: Ticker Input ───────────────────────────────────────────────────
st.subheader("Step 1 — Enter a Stock Ticker")
col1, col2 = st.columns([2, 1])
with col1:
    ticker = st.text_input("Stock ticker symbol", placeholder="e.g. AAPL, MSFT, NVDA",
                           max_chars=6).upper().strip()
with col2:
    fetch_btn = st.button("🔍 Fetch Live Data", use_container_width=True,
                          disabled=not (ticker and api_key))

# ── Fetch stock data via Claude ────────────────────────────────────────────
def fetch_stock_data(ticker: str, key: str) -> dict:
    client = Anthropic(api_key=key)
    with st.spinner(f"Fetching live data for {ticker}..."):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=(
                "You are a financial data assistant. Search the web for the requested "
                "stock ticker and return ONLY a valid JSON object with no markdown, "
                "no backticks, no preamble. Required keys:\n"
                "- companyName (string)\n"
                "- currentPrice (number, USD)\n"
                "- marketCap (string, e.g. '2.8T')\n"
                "- revenue (number, billions USD, trailing 12 months)\n"
                "- netMargin (number, percentage e.g. 25.3)\n"
                "- sharesOutstanding (number, millions)\n"
                "- peRatio (number or null)\n"
                "- analystTarget (number or null)\n"
                "- sector (string)\n"
                "Use null for unavailable values. Return ONLY the JSON object."
            ),
            messages=[{"role": "user",
                        "content": f"Get current financial data for stock ticker: {ticker}"}]
        )
    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text
    text = text.strip().replace("```json", "").replace("```", "").strip()
    import json, re
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group())
        raise ValueError("Could not parse stock data response.")

# ── Session state ──────────────────────────────────────────────────────────
if "stock" not in st.session_state:
    st.session_state.stock = None
if "dcf_run" not in st.session_state:
    st.session_state.dcf_run = False

if fetch_btn and ticker and api_key:
    try:
        st.session_state.stock = fetch_stock_data(ticker, api_key)
        st.session_state.stock["ticker"] = ticker
        st.session_state.dcf_run = False
    except Exception as e:
        st.error(f"Could not fetch data: {e}")

# ── Display fetched data ───────────────────────────────────────────────────
if st.session_state.stock:
    sd = st.session_state.stock
    st.success(f"✅ Data loaded for **{sd.get('companyName', ticker)}** ({ticker})")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    price = sd.get("currentPrice")
    target = sd.get("analystTarget")
    upside = ((target - price) / price * 100) if price and target else None

    c1.metric("Current Price",    f"${price:.2f}"                      if price   else "—")
    c2.metric("Market Cap",       sd.get("marketCap", "—"))
    c3.metric("TTM Revenue",      f"${sd['revenue']:.1f}B"             if sd.get("revenue")          else "—")
    c4.metric("Net Margin",       f"{sd['netMargin']:.1f}%"            if sd.get("netMargin")        else "—")
    c5.metric("P/E Ratio",        f"{sd['peRatio']:.1f}x"             if sd.get("peRatio")          else "—")
    c6.metric("Analyst Target",   f"${target:.2f}",
              delta=f"{upside:+.1f}%" if upside is not None else None)

    st.markdown("---")

    # ── Step 2: Assumptions ────────────────────────────────────────────────
    st.subheader("Step 2 — Valuation Assumptions")
    st.caption("Adjust the sliders below. Hover over ℹ️ labels for explanations.")

    col_a, col_b = st.columns(2)
    with col_a:
        growth_rate = st.slider(
            "📈 Revenue growth rate — years 1–5 (%)",
            0.0, 40.0, 10.0, 0.5,
            help="How fast you expect revenue to grow in the near term. "
                 "Compare to analyst consensus or the company's historical CAGR."
        )
        term_growth = st.slider(
            "🌱 Terminal growth rate — years 6–10 (%)",
            0.0, 6.0, 3.0, 0.25,
            help="Long-run sustainable growth rate. Typically close to GDP growth (2–4%). "
                 "Above 5% is very aggressive and rarely justified."
        )
        wacc = st.slider(
            "💰 Discount rate / WACC (%)",
            5.0, 20.0, 10.0, 0.25,
            help="Weighted Average Cost of Capital — your required rate of return. "
                 "8–12% is typical for large-cap US stocks. Higher = riskier company."
        )

    with col_b:
        default_margin = round(sd.get("netMargin", 20.0), 1)
        margin = st.slider(
            "💵 Net profit margin (%)",
            1.0, 50.0, float(min(50, max(1, default_margin))), 0.5,
            help="Net income as % of revenue. Pre-filled from live data. "
                 "Adjust if you expect margin expansion or compression going forward."
        )
        pe_exit = st.slider(
            "🔢 P/E exit multiple at year 10 (x)",
            8, 40, 18, 1,
            help="The earnings multiple applied at the end of Year 10 to estimate terminal value. "
                 "Use the sector median as a baseline. Higher = more optimistic."
        )
        default_shares = round(sd.get("sharesOutstanding", 1000))
        shares = st.slider(
            "🔷 Shares outstanding (millions)",
            50, 30000, int(min(30000, max(50, default_shares))), 50,
            help="Auto-filled from live data. Decrease for buyback-heavy companies; "
                 "increase if dilution is expected (stock-based compensation, equity raises)."
        )

    run_btn = st.button("▶️ Run DCF Model", use_container_width=True, type="primary")

    # ── Step 3: DCF Calculation ────────────────────────────────────────────
    if run_btn or st.session_state.dcf_run:
        st.session_state.dcf_run = True
        base_revenue = sd.get("revenue", 0)
        if not base_revenue:
            st.error("Revenue data is missing — cannot run DCF.")
            st.stop()

        g1   = growth_rate / 100
        gT   = term_growth / 100
        disc = wacc / 100
        marg = margin / 100

        rows = []
        rev = base_revenue
        for y in range(1, 11):
            rate = g1 if y <= 5 else g1 + (gT - g1) * ((y - 5) / 5)
            rev  = rev * (1 + rate)
            earn = rev * marg
            pv   = earn / (1 + disc) ** y
            rows.append({"Year": y, "Revenue ($B)": rev, "Net Earnings ($B)": earn,
                          "Discount Factor": 1/(1+disc)**y, "PV of Earnings ($B)": pv,
                          "Growth Rate (%)": rate*100})

        df = pd.DataFrame(rows)
        term_earn   = df.iloc[-1]["Net Earnings ($B)"]
        term_value  = term_earn * pe_exit
        pv_terminal = term_value / (1 + disc) ** 10
        pv_cf       = df["PV of Earnings ($B)"].sum()
        total_val   = pv_cf + pv_terminal
        intrinsic   = (total_val * 1e9) / (shares * 1e6)
        current     = sd.get("currentPrice")
        mos         = ((intrinsic - current) / intrinsic * 100) if current else None
        term_pct    = pv_terminal / total_val * 100

        st.markdown("---")
        st.subheader("Step 3 — Model Breakdown")

        # Step cards
        st.markdown(f"""
<div class="step-box">
<b>Step 1 of 4 — Project revenue & earnings (years 1–10)</b><br><br>
Starting from <b>${base_revenue:.1f}B</b> in trailing revenue, we apply your growth assumptions.
Years 1–5 grow at <b>{growth_rate:.1f}%</b>, then linearly taper toward the terminal rate of
<b>{term_growth:.1f}%</b> by year 10. Net earnings each year = Revenue × <b>{margin:.1f}%</b> margin.
</div>
""", unsafe_allow_html=True)

        fmt_df = df.copy()
        fmt_df["Revenue ($B)"]       = fmt_df["Revenue ($B)"].map("${:.2f}B".format)
        fmt_df["Net Earnings ($B)"]  = fmt_df["Net Earnings ($B)"].map("${:.2f}B".format)
        fmt_df["Discount Factor"]    = fmt_df["Discount Factor"].map("{:.4f}".format)
        fmt_df["PV of Earnings ($B)"]= fmt_df["PV of Earnings ($B)"].map("${:.2f}B".format)
        fmt_df["Growth Rate (%)"]    = fmt_df["Growth Rate (%)"].map("{:.2f}%".format)
        st.dataframe(fmt_df.set_index("Year"), use_container_width=True)

        st.markdown(f"""
<div class="step-box">
<b>Step 2 of 4 — Discount earnings to present value</b><br><br>
A dollar earned in the future is worth less today due to risk and opportunity cost.
We divide each year's earnings by <b>(1 + {wacc:.1f}%)^year</b>.
For example, Year 10 earnings of <b>${df.iloc[-1]['Net Earnings ($B)']:.2f}B</b>
discount to only <b>${df.iloc[-1]['PV of Earnings ($B)']:.2f}B</b> in today's dollars.<br><br>
<b>Sum of all discounted earnings = ${pv_cf:.2f}B</b>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="step-box">
<b>Step 3 of 4 — Calculate terminal value</b><br><br>
At Year 10, we apply a <b>{pe_exit}x P/E exit multiple</b> to Year 10 earnings of
<b>${term_earn:.2f}B</b> → Terminal value = <b>${term_value:.2f}B</b>.<br>
Discounted back 10 years at {wacc:.1f}% → <b>PV of terminal value = ${pv_terminal:.2f}B</b>.<br><br>
⚠️ Terminal value = <b>{term_pct:.1f}% of total estimated value</b>.
This shows how sensitive DCF is to the exit multiple and terminal growth assumptions.
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="step-box">
<b>Step 4 of 4 — Intrinsic value per share</b><br><br>
Total enterprise value = PV of cash flows (<b>${pv_cf:.2f}B</b>) + PV of terminal value (<b>${pv_terminal:.2f}B</b>)
= <b>${total_val:.2f}B</b><br><br>
Per share: ${total_val:.2f}B ÷ {shares}M shares = <b>${intrinsic:.2f}</b>
</div>
""", unsafe_allow_html=True)

        # ── Charts ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Visual Breakdown")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=df["Year"], y=df["Revenue ($B)"], name="Revenue", marker_color="#90caf9"))
            fig1.add_trace(go.Bar(
                x=df["Year"], y=df["Net Earnings ($B)"], name="Net Earnings", marker_color="#1976d2"))
            fig1.update_layout(title="Revenue & Earnings Projection",
                               barmode="overlay", xaxis_title="Year",
                               yaxis_title="$B", height=320, margin=dict(t=40, b=20))
            st.plotly_chart(fig1, use_container_width=True)

        with chart_col2:
            labels = [f"Yr {r['Year']} Earnings" for _, r in df.iterrows()] + ["Terminal Value"]
            values = list(df["PV of Earnings ($B)"]) + [pv_terminal]
            colors = ["#bbdefb"] * 10 + ["#1976d2"]
            fig2 = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))
            fig2.update_layout(title=f"Value Composition (Terminal = {term_pct:.1f}%)",
                               xaxis_title="Component", yaxis_title="PV ($B)",
                               height=320, margin=dict(t=40, b=20),
                               xaxis=dict(tickangle=-45))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Verdict ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🏁 Valuation Verdict")

        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Intrinsic Value", f"${intrinsic:.2f}")
        if current:
            v_col2.metric("Market Price", f"${current:.2f}")
            v_col3.metric("Margin of Safety", f"{mos:+.1f}%",
                          delta_color="normal" if mos and mos > 0 else "inverse")

        if mos is not None:
            if mos > 20:
                cls, label, icon = "undervalued", "Potentially Undervalued", "🟢"
            elif mos < -15:
                cls, label, icon = "overvalued", "Potentially Overvalued", "🔴"
            else:
                cls, label, icon = "fair", "Fairly Valued", "🟡"

            st.markdown(f"""
<div class="verdict-box {cls}">
<h3>{icon} {label}</h3>
<p>Your DCF model estimates intrinsic value at <b>${intrinsic:.2f}</b> vs. the current market price of
<b>${current:.2f}</b> — a margin of safety of <b>{mos:+.1f}%</b>.</p>
<p>A margin of safety &gt;20% is often used as a buy signal by value investors to account for model uncertainty.
A negative margin suggests the market expects more growth than your assumptions model.</p>
</div>
""", unsafe_allow_html=True)

        # ── Sensitivity table ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔬 Sensitivity Analysis")
        st.caption("Intrinsic value per share across different WACC and growth rate combinations")

        wacc_range   = [wacc - 2, wacc - 1, wacc, wacc + 1, wacc + 2]
        growth_range = [growth_rate - 3, growth_rate - 1.5, growth_rate,
                        growth_rate + 1.5, growth_rate + 3]

        sens_data = {}
        for g in growth_range:
            row = {}
            for w in wacc_range:
                g_   = max(0, g) / 100
                w_   = max(1, w) / 100
                r2   = base_revenue
                pvs  = 0
                for y in range(1, 11):
                    rate2 = g_ if y <= 5 else g_ + (gT - g_) * ((y-5)/5)
                    r2   *= (1 + rate2)
                    pvs  += r2 * marg / (1 + w_) ** y
                te  = r2 * marg
                tv2 = te * pe_exit / (1 + w_) ** 10
                iv  = ((pvs + tv2) * 1e9) / (shares * 1e6)
                row[f"WACC {w:.1f}%"] = f"${iv:.2f}"
            sens_data[f"Growth {g:.1f}%"] = row

        sens_df = pd.DataFrame(sens_data).T
        st.dataframe(sens_df, use_container_width=True)
        st.caption("Values in bold-ish column represent your current assumption. Vary both axes to understand the range of outcomes.")

else:
    st.info("👆 Enter a ticker symbol and your Anthropic API key in the sidebar, then click **Fetch Live Data** to begin.")
# coding: utf-8 -*-
"""
Spyder Editor
 -*-
This is a temporary script file.
"""

