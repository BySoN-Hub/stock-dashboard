import yfinance as yf
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="米国株 スコアランキング", layout="wide")

# ===== パスワード保護 =====
def check_password():
    def password_entered():
        if st.session_state["pw"] == st.secrets["password"]:
            st.session_state["ok"] = True
            del st.session_state["pw"]
        else:
            st.session_state["ok"] = False
    if st.session_state.get("ok", False):
        return True
    st.text_input("パスワードを入力してください", type="password",
                  on_change=password_entered, key="pw")
    if "ok" in st.session_state and not st.session_state["ok"]:
        st.error("パスワードが違います")
    return False

if not check_password():
    st.stop()
# ===== ここまで =====

st.title("米国株 テクニカル スコアランキング")
st.caption("判断材料の補助ツールです。スコアやシグナルは指標を点数化したもので、上がる確率や売買推奨ではありません。")

INDEX_OPTIONS = {
    "NASDAQ100（米ハイテク中心100社）": "nasdaq100",
    "S&P500（米国の主要500社）": "sp500",
    "ダウ平均（米国の代表30社）": "dowjones",
}

@st.cache_data(ttl=86400)
def load_symbols(index_code):
    url = f"https://yfiua.github.io/index-constituents/constituents-{index_code}.csv"
    df = pd.read_csv(url)
    df.columns = [c.lower() for c in df.columns]
    return df[["symbol", "name"]].dropna()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=1800)
def load_raw(index_code):
    syms = load_symbols(index_code)
    tickers = syms["symbol"].tolist()
    name_map = dict(zip(syms["symbol"], syms["name"]))
    raw = yf.download(tickers, period="6mo", interval="1d",
                      progress=False, auto_adjust=True, group_by="ticker")
    fetched_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    return raw, tickers, name_map, fetched_at

def build_scores(raw, tickers, name_map):
    rows = []
    for t in tickers:
        try:
            close = raw[t]["Close"].dropna()
            volume = raw[t]["Volume"].dropna()
        except Exception:
            continue
        if len(close) < 80:
            continue
        ma5  = float(close.rolling(5).mean().iloc[-1])
        ma25 = float(close.rolling(25).mean().iloc[-1])
        ma75 = float(close.rolling(75).mean().iloc[-1])
        rsi  = float(calc_rsi(close).iloc[-1])
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_s = ema12 - ema26
        signal_s = macd_s.ewm(span=9).mean()
        hist_s = macd_s - signal_s
        v_macd, v_signal = float(macd_s.iloc[-1]), float(signal_s.iloc[-1])
        v_hist, hist_prev = float(hist_s.iloc[-1]), float(hist_s.iloc[-2])
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        recent_high = float(close.tail(60).max())
        recent_low  = float(close.tail(60).min())
        dev25 = (last_close - ma25) / ma25 * 100

        chg = (last_close - prev_close) / prev_close * 100

        vol_avg = float(volume.tail(25).mean())
        vol_last = float(volume.iloc[-1])
        vol_ratio = vol_last / vol_avg if vol_avg > 0 else 0

        s = {}
        if rsi <= 30:   s["RSI"] = 20
        elif rsi <= 40: s["RSI"] = 16
        elif rsi <= 55: s["RSI"] = 12
        elif rsi <= 70: s["RSI"] = 7
        else:           s["RSI"] = 3
        if ma5 > ma25 > ma75: s["trend"] = 20
        elif ma25 > ma75:     s["trend"] = 14
        elif ma5 > ma25:      s["trend"] = 10
        else:                 s["trend"] = 4
        if ma25 > ma75 and -8 <= dev25 <= -1: s["dip"] = 20
        elif ma25 > ma75 and -1 < dev25 <= 3: s["dip"] = 13
        elif dev25 <= -8:                      s["dip"] = 9
        elif dev25 >= 8:                       s["dip"] = 4
        else:                                  s["dip"] = 10
        if v_macd > v_signal and v_hist > hist_prev:  s["macd"] = 20
        elif v_macd > v_signal:                        s["macd"] = 14
        elif v_macd < v_signal and v_hist > hist_prev: s["macd"] = 11
        else:                                          s["macd"] = 5
        pos = (last_close - recent_low) / (recent_high - recent_low + 1e-9) * 100
        if pos <= 25:   s["pos"] = 20
        elif pos <= 50: s["pos"] = 14
        elif pos <= 75: s["pos"] = 9
        else:           s["pos"] = 4
        buy_total = sum(s.values())

        sig = []
        if rsi >= 75 or dev25 >= 12:
            sig.append("過熱注意")
        if rsi <= 30 and ma25 > ma75:
            sig.append("押し目候補")
        if vol_ratio >= 2:
            sig.append("出来高急増")
        signal_text = " / ".join(sig) if sig else "—"

        rows.append({
            "ティッカー": t, "銘柄": name_map.get(t, t),
            "終値$": round(last_close, 2),
            "前日比%": round(chg, 2),
            "RSI": round(rsi, 1),
            "25日線乖離%": round(dev25, 1),
            "出来高倍率": round(vol_ratio, 1),
            "シグナル": signal_text,
            "RSIスコア": s["RSI"], "トレンドスコア": s["trend"],
            "押し目スコア": s["dip"], "MACDスコア": s["macd"],
            "価格位置スコア": s["pos"], "総合スコア": buy_total,
        })
    return pd.DataFrame(rows)

index_label = st.selectbox("対象とする市場（指数）を選択", list(INDEX_OPTIONS.keys()))
index_code = INDEX_OPTIONS[index_label]

with st.spinner(f"{index_label} を取得・計算中…（銘柄数により1〜数分）"):
    raw, tickers, name_map, fetched_at = load_raw(index_code)
    df = build_scores(raw, tickers, name_map)

df = df.sort_values("総合スコア", ascending=False).reset_index(drop=True)
df.index = df.index + 1
st.success(f"{len(df)} 銘柄を分析しました")
st.info(f"データ最終取得：{fetched_at.strftime('%Y年%m月%d日 %H:%M')}（日本時間） ※株価は15〜20分遅れの遅延データ")

col_a, col_b = st.columns(2)
with col_a:
    top_n = st.slider("表示する上位件数", 10, len(df), 30, step=5)
with col_b:
    view_mode = st.selectbox(
        "目的で絞り込む",
        ["すべて表示", "買い検討（押し目候補）", "売り検討（過熱注意）", "出来高急増"]
    )

view = df.copy()
if view_mode == "買い検討（押し目候補）":
    view = view[view["シグナル"].str.contains("押し目候補")]
    view = view.sort_values("総合スコア", ascending=False)
elif view_mode == "売り検討（過熱注意）":
    view = view[view["シグナル"].str.contains("過熱注意")]
    # 売り目線では「より過熱しているもの」が上に来るようRSIの高い順に
    view = view.sort_values("RSI", ascending=False)
elif view_mode == "出来高急増":
    view = view[view["シグナル"].str.contains("出来高急増")]
    view = view.sort_values("出来高倍率", ascending=False)

view = view.head(top_n)

if view.empty:
    st.warning("この条件に当てはまる銘柄は今はありません。")


st.subheader("総合スコアランキング")
def color_chg(v):
    if v > 0: return "color: green"
    if v < 0: return "color: red"
    return ""

# 表で銘柄を選べるようにする（行選択を有効化）
event = st.dataframe(
    view.style
        .background_gradient(subset=["総合スコア"], cmap="RdYlGn", vmin=40, vmax=100)
        .bar(subset=["RSIスコア","トレンドスコア","押し目スコア","MACDスコア","価格位置スコア"],
             color="#9ad0ec", vmin=0, vmax=20)
        .map(color_chg, subset=["前日比%"])
        .format({"終値$":"{:.2f}", "前日比%":"{:+.2f}", "RSI":"{:.1f}",
                 "25日線乖離%":"{:+.1f}", "出来高倍率":"{:.1f}倍"}),
    use_container_width=True, height=600,
    on_select="rerun", selection_mode="single-row",
    key="rank_table",
)

st.subheader("上位3銘柄")
top3 = view.head(3).reset_index()
cols = st.columns(3)
medals = ["1位", "2位", "3位"]
for i, (_, r) in enumerate(top3.iterrows()):
    with cols[i]:
        st.metric(f"{medals[i]}　{r['ティッカー']}",
                  f"{r['総合スコア']} 点", f"{r['前日比%']:+.2f}%")
        st.progress(int(r["総合スコア"]) / 100)
        st.caption(f"シグナル: {r['シグナル']}")

st.divider()
st.subheader("銘柄チャート（株価＋移動平均線＋RSI）")

# 表で行が選ばれていれば、その銘柄を使う。なければプルダウンで選択。
sel_ticker = None
selected_rows = event.selection.rows if event and event.selection else []
if selected_rows:
    sel_ticker = view.iloc[selected_rows[0]]["ティッカー"]
    st.caption(f"表で選択中：{sel_ticker}")
else:
    choices = [f"{r['ティッカー']}　{r['銘柄']}" for _, r in df.iterrows()]
    selected = st.selectbox("チャートを見たい銘柄を選択（表の行をクリックしても切替わります）", choices)
    sel_ticker = selected.split("　")[0]

close = raw[sel_ticker]["Close"].dropna()
ma5  = close.rolling(5).mean()
ma25 = close.rolling(25).mean()
ma75 = close.rolling(75).mean()
rsi  = calc_rsi(close)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3], vertical_spacing=0.05,
                    subplot_titles=(f"{sel_ticker} 株価と移動平均線", "RSI"))
fig.add_trace(go.Scatter(x=close.index, y=close, name="終値", line=dict(color="black")), row=1, col=1)
fig.add_trace(go.Scatter(x=ma5.index,  y=ma5,  name="5日線",  line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=ma25.index, y=ma25, name="25日線", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=ma75.index, y=ma75, name="75日線", line=dict(color="green")), row=1, col=1)
fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="purple")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red",  row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(height=600, hovermode="x unified", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("構成銘柄は yfiua/index-constituents、株価は yfinance の遅延データ。スコア・シグナルは一例で上がる確率ではありません。最終判断はご自身で。")
