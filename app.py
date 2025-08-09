import os
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Tushare SDK
try:
    import tushare as ts
except Exception:
    ts = None

st.set_page_config(page_title="A股一键分析 | 工作流", page_icon="📈", layout="wide")

# ----------------------- Helpers -----------------------
def detect_market(code: str) -> str:
    code = code.strip().upper()
    if code.endswith((".SH", ".SZ", ".BJ")):
        return code
    if code.startswith("6"):
        return f"{code}.SH"
    if code.startswith(("0","3")):
        return f"{code}.SZ"
    if code.startswith(("4","8")):
        return f"{code}.BJ"
    return code

def atr(series_high, series_low, series_close, n=14):
    high = series_high.astype(float)
    low = series_low.astype(float)
    close = series_close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def sma(series, n):
    return series.rolling(n).mean()

def roc(series, n=20):
    return series.pct_change(n)

def score_technicals(df):
    s = 0
    reasons = []
    # Price vs MA200
    if df["close"].iloc[-1] > df["ma200"].iloc[-1]:
        s += 20; reasons.append("价>MA200 +20")
    else:
        s -= 20; reasons.append("价<MA200 -20")
    # MA50 vs MA200
    if df["ma50"].iloc[-1] > df["ma200"].iloc[-1]:
        s += 20; reasons.append("MA50>MA200 +20")
    else:
        s -= 10; reasons.append("MA50<MA200 -10")
    # MA20 vs MA50
    if df["ma20"].iloc[-1] > df["ma50"].iloc[-1]:
        s += 15; reasons.append("MA20>MA50 +15")
    else:
        s -= 10; reasons.append("MA20<MA50 -10")
    # Momentum
    if df["roc20"].iloc[-1] > 0:
        s += 15; reasons.append("动量向上 +15")
    else:
        s -= 15; reasons.append("动量转弱 -15")
    # 52w high/low proximity
    close = df["close"].iloc[-1]
    high52 = df["high_52w"].iloc[-1]
    low52 = df["low_52w"].iloc[-1]
    if pd.notna(high52) and (high52 - close) / close <= 0.05:
        s += 10; reasons.append("接近52周高 +10")
    if pd.notna(low52) and (close - low52) / close <= 0.05:
        s -= 10; reasons.append("接近52周低 -10")
    # Volatility
    atrp = df["atr14"].iloc[-1] / close
    if atrp <= 0.03:
        s += 10; reasons.append("低波动 +10")
    elif atrp >= 0.06:
        s -= 10; reasons.append("高波动 -10")
    else:
        reasons.append("中等波动 0")
    return int(s), reasons, float(atrp)

def action_from_score(score, close, ma200):
    if score >= 30 and close > ma200:
        return "买入（趋势多头）"
    if 10 <= score < 30:
        return "分批/观察"
    if -10 < score < 10:
        return "观望"
    if score <= -30 and close < ma200:
        return "回避/减仓"
    return "谨慎"

def mk_ts_pro():
    token = st.secrets.get("TUSHARE_TOKEN", os.environ.get("TUSHARE_TOKEN", ""))
    if not token:
        st.warning("未检测到 Tushare Token。请在侧边栏输入，或在 Streamlit Secrets 中配置 TUSHARE_TOKEN。")
        return None, None
    try:
        ts.set_token(token)  # ensure pro_bar reads it
    except Exception:
        pass
    pro = ts.pro_api(token) if ts else None
    return token, pro

def fetch_daily(pro, ts_code, start_date):
    try:
        if isinstance(start_date, dt.date):
            start = start_date.strftime("%Y%m%d")
        else:
            start = str(start_date).replace("-", "")
        end = dt.date.today().strftime("%Y%m%d")

        import tushare as ts2
        # primary: qfq day bars
        df = ts2.pro_bar(ts_code=ts_code, adj='qfq', asset='E', freq='D', start_date=start, end_date=end)
        # fallback: pro.daily
        if df is None or df.empty:
            df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.sort_values("trade_date")
            df["date"] = pd.to_datetime(df["trade_date"])
            # unify columns
            rename_map = {"open":"open","high":"high","low":"low","close":"close","vol":"volume","volume":"volume"}
            for k,v in list(rename_map.items()):
                if k not in df.columns and k=="vol" and "vol" not in df.columns and "volume" in df.columns:
                    # already has volume
                    pass
            if "volume" not in df.columns:
                # some endpoints give 'vol'
                if "vol" in df.columns:
                    df["volume"] = df["vol"]
                else:
                    df["volume"] = np.nan
        else:
            df = df.sort_values("trade_date")
            df["date"] = pd.to_datetime(df["trade_date"])
            if "vol" in df.columns and "volume" not in df.columns:
                df = df.rename(columns={"vol":"volume"})
        return df[["date","open","high","low","close","volume"]].reset_index(drop=True)
    except Exception as e:
        st.error(f"拉取日线失败：{e}")
        return pd.DataFrame()

def fetch_basics(pro, ts_code):
    try:
        for i in range(0,5):
            day = (dt.date.today() - dt.timedelta(days=i)).strftime("%Y%m%d")
            dfb = pro.daily_basic(ts_code=ts_code, trade_date=day, fields="ts_code,pe,pe_ttm,pb,ps_ttm,total_mv,circ_mv,turnover_rate,close")
            if dfb is not None and not dfb.empty:
                return dfb
        return pd.DataFrame()
    except Exception as e:
        st.info(f"估值数据暂不可用：{e}")
        return pd.DataFrame()

def compute_indicators(df):
    out = df.copy()
    out["ma20"] = sma(out["close"], 20)
    out["ma50"] = sma(out["close"], 50)
    out["ma200"] = sma(out["close"], 200)
    out["roc20"] = roc(out["close"], 20)
    out["atr14"] = atr(out["high"], out["low"], out["close"], 14)
    out["high_52w"] = out["close"].rolling(252).max()
    out["low_52w"] = out["close"].rolling(252).min()
    return out

def score_fundamentals(pe_quantile):
    if pd.isna(pe_quantile):
        return 0, "估值分位未知 0"
    if pe_quantile < 0.4:
        return 10, "PE分位<40% +10"
    if pe_quantile > 0.7:
        return -10, "PE分位>70% -10"
    return 0, "PE分位40%~70% 0"

def combine_scores(tech_score, industry_score, company_score, val_score, policy_score, weight_tech=0.6):
    other = industry_score + company_score + val_score + policy_score  # max 65
    other_norm = (other / 65.0) * 100.0
    final = int(round(tech_score * weight_tech + other_norm * (1 - weight_tech)))
    return final

# ----------------------- UI -----------------------
st.title("📈 A股一键分析 · 在线版（V1）")

with st.sidebar:
    st.header("参数")
    code_input = st.text_input("股票代码", value="600519", help="可填 600519 或 600519.SH/000001.SZ 等格式")
    start_date = st.date_input("起始日期", value=dt.date.today()-dt.timedelta(days=700))
    risk_pct = st.number_input("单笔风险预算（占总资金）", min_value=0.002, max_value=0.05, value=0.01, step=0.002)
    weight_tech = st.slider("技术面权重", 0.3, 0.8, 0.6, 0.05)

    st.subheader("行业/政策（V1 手动评估，未来接入自动化）")
    industry_score = st.slider("行业分（-30~+30）", -30, 30, 10)
    company_score = st.slider("公司相对分（-15~+15）", -15, 15, 5)
    policy_score = st.slider("政策分（-10~+10）", -10, 10, 4)
    val_quantile = st.slider("估值PE分位（0~1，越低越便宜）", 0.0, 1.0, 0.45, 0.05)

    st.caption("提示：行业/政策为参考打分，建议结合渠道与研报。")

    token_manual = st.text_input("Tushare Token（留空则使用Secrets/环境变量）", value="")

if ts is None:
    st.error("未安装 tushare，请在 requirements.txt 中包含 tushare。")
else:
    if token_manual:
        ts.set_token(token_manual)

token, pro = mk_ts_pro()
ts_code = detect_market(code_input)

tab1, tab2 = st.tabs(["🔍 分析", "⚙️ 说明与方法"])

with tab1:
    st.write(f"**标的：** {ts_code}")

    if pro is None:
        st.stop()

    df = fetch_daily(pro, ts_code, start_date)
    if df.empty:
        st.warning("未获取到行情数据，请确认代码是否正确、Token是否有效。")
        st.stop()

    di = compute_indicators(df).dropna().reset_index(drop=True)
    if di.empty or len(di) < 210:
        st.warning("可用数据不足以计算长期指标（至少需要200+交易日）。")
    latest = di.iloc[-1]

    tech_score, tech_reasons, atrp = score_technicals(di)
    val_score, val_reason = score_fundamentals(val_quantile)

    final = combine_scores(
        tech_score=tech_score,
        industry_score=industry_score,
        company_score=company_score,
        val_score=val_score,
        policy_score=policy_score,
        weight_tech=weight_tech
    )

    col1, col2 = st.columns([2,1], gap="large")
    with col1:
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(di["date"], di["close"], label="Close")
        ax.plot(di["date"], di["ma20"], label="MA20")
        ax.plot(di["date"], di["ma50"], label="MA50")
        ax.plot(di["date"], di["ma200"], label="MA200")
        ax.set_title(f"{ts_code} 价格与均线")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("指标打分")
        st.metric("技术面总分", tech_score)
        st.write("· " + "；".join(tech_reasons))
        st.write(f"估值：{val_reason}")
        st.write(f"行业：{industry_score}；公司相对：{company_score}；政策：{policy_score}")

        action = action_from_score(tech_score, latest['close'], latest['ma200'])
        atr_val = latest.get("atr14", np.nan)
        stop_loss = latest["close"] - 2 * atr_val if pd.notna(atr_val) else np.nan
        pos_coef = None
        if pd.notna(stop_loss) and stop_loss < latest["close"]:
            pos_coef = (10000 * risk_pct) / (latest["close"] - stop_loss)

        st.markdown("---")
        st.subheader("操作建议（基于技术面）")
        st.write(f"**建议**：{action}")
        st.write(f"**止损价**：{stop_loss:.2f}" if pd.notna(stop_loss) else "止损价：N/A")
        if pos_coef:
            st.write(f"**建议仓位系数**：每 1 万资金买 **{pos_coef:.0f} 股**")
        st.caption("仓位计算：单笔最大亏损≤资金×风险预算；止损=收盘-2×ATR14")

    st.markdown("### 综合判断")
    st.write(f"**综合评分（技术{int(weight_tech*100)}% + 基本面/行业/政策 {int((1-weight_tech)*100)}%）**：{final}")
    if final >= 70:
        st.success("综合结论：分批买入")
    elif final >= 50:
        st.info("综合结论：观察/轻仓")
    elif final >= 30:
        st.warning("综合结论：观望")
    else:
        st.error("综合结论：回避/减仓")

    basics = fetch_basics(pro, ts_code)
    if basics is not None and not basics.empty:
        st.markdown("#### 估值快照（最近交易日）")
        st.dataframe(basics)

with tab2:
    st.markdown("""
**方法概览**  
- 行情：Tushare `pro_bar` 复权日线；指标：MA20/50/200、ROC20、ATR14、52周高低  
- 技术打分：趋势>动量>波动权重；-100~+100  
- 行业/政策：V1手动评分（未来接数据源自动化）  
- 估值：以PE分位映射（<40% +10；>70% -10）  
- 综合：`final = 技术×60% + 其它（行业+公司+估值+政策）标准化×40%`  
- 风险控制：`止损 = 收盘 - 2×ATR14`；`仓位 = 资金×风险% / (入场-止损)`

**使用提示**  
- 代码可填 `600519` 或 `600519.SH`。  
- 若首次使用，请在侧栏填入 Tushare Token，或在部署平台的 Secrets 中设置 `TUSHARE_TOKEN`。  
- 本工具为研究参考，不构成投资建议。
""")