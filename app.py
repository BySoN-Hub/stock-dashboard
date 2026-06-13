import yfinance as yf, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from zoneinfo import ZoneInfo

st.set_page_config(page_title="米国株 スコアランキング", layout="wide")

# ===== スマホ向け CSS =====
st.markdown("""
<style>
html, body, .stApp, p, div, span, table, th, td {
    font-family: 'Meiryo', 'メイリオ', sans-serif !important;
}
.block-container {padding: 2.2rem 0.6rem 1rem 0.6rem !important;}
h1 {font-size: 1.35rem !important; margin-top: 0.2rem !important; margin-bottom: 0.3rem !important; line-height: 1.3 !important;}
h2, h3 {font-size: 1.05rem !important; margin-top: 0.5rem !important; margin-bottom: 0.3rem !important;}
.stDataFrame {font-size: 0.78rem !important;}
.stCaption, .caption {font-size: 0.72rem !important;}
div[data-testid="stVerticalBlock"] {gap: 0.4rem !important;}
hr {margin: 0.4rem 0 !important;}
label {font-size: 0.82rem !important;}
</style>
""", unsafe_allow_html=True)

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
    raw = yf.download(tickers, period="6mo", interval="1d", progress=False,
                      auto_adjust=True, group_by="ticker")
    fetched_at = datetime.now(ZoneInfo("Asia/Tokyo"))
    return raw, tickers, name_map, fetched_at

def score_from_series(close, volume):
    if len(close) < 80:
        return None
    ma5, ma25, ma75 = float(close.rolling(5).mean().iloc[-1]), float(close.rolling(25).mean().iloc[-1]), float(close.rolling(75).mean().iloc[-1])
    rsi = float(calc_rsi(close).iloc[-1])
    ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    v_macd, v_signal = float(macd.iloc[-1]), float(signal.iloc[-1])
    v_hist, hist_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    last_close = float(close.iloc[-1])
    recent_high, recent_low = float(close.tail(60).max()), float(close.tail(60).min())
    dev25 = (last_close - ma25) / ma25 * 100
    vol_avg = float(volume.tail(25).mean())
    vol_last = float(volume.iloc[-1])
    vol_ratio = vol_last / vol_avg if vol_avg > 0 else 0

    s = {}
    s["RSI"] = 20 if rsi <= 30 else 16 if rsi <= 40 else 12 if rsi <= 55 else 7 if rsi <= 70 else 3
    s["trend"] = 20 if ma5 > ma25 > ma75 else 14 if ma25 > ma75 else 10 if ma5 > ma25 else 4
    if ma25 > ma75 and -8 <= dev25 <= -1:
        s["dip"] = 20
    elif ma25 > ma75 and -1 < dev25 <= 3:
        s["dip"] = 13
    elif dev25 <= -8:
        s["dip"] = 9
    elif dev25 >= 8:
        s["dip"] = 4
    else:
        s["dip"] = 10
    if v_macd > v_signal and v_hist > hist_prev:
        s["macd"] = 20
    elif v_macd > v_signal:
        s["macd"] = 14
    elif v_macd < v_signal and v_hist > hist_prev:
        s["macd"] = 11
    else:
        s["macd"] = 5
    pos = (last_close - recent_low) / (recent_high - recent_low + 1e-9) * 100
    s["pos"] = 20 if pos <= 25 else 14 if pos <= 50 else 9 if pos <= 75 else 4
    total = sum(s.values())
    return {"total": total, "rsi": rsi, "dev25": dev25, "vol_ratio": vol_ratio,
            "ma5": ma5, "ma25": ma25, "ma75": ma75,
            "v_macd": v_macd, "v_signal": v_signal,
            "recent_high": recent_high, "recent_low": recent_low,
            "last_close": last_close}

def build_scores_full(raw, tickers, name_map):
    rows = []
    for t in tickers:
        try:
            close = raw[t]["Close"].dropna()
            volume = raw[t]["Volume"].dropna()
        except Exception:
            continue
        if len(close) < 80:
            continue
        ma5, ma25, ma75 = float(close.rolling(5).mean().iloc[-1]), float(close.rolling(25).mean().iloc[-1]), float(close.rolling(75).mean().iloc[-1])
        rsi = float(calc_rsi(close).iloc[-1])
        ema12, ema26 = close.ewm(span=12).mean(), close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal
        v_macd, v_signal = float(macd.iloc[-1]), float(signal.iloc[-1])
        v_hist, hist_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
        last_close, prev_close = float(close.iloc[-1]), float(close.iloc[-2])
        recent_high, recent_low = float(close.tail(60).max()), float(close.tail(60).min())
        dev25 = (last_close - ma25) / ma25 * 100
        chg = (last_close - prev_close) / prev_close * 100
        vol_avg = float(volume.tail(25).mean())
        vol_last = float(volume.iloc[-1])
        vol_ratio = vol_last / vol_avg if vol_avg > 0 else 0

        s = {}
        s["RSI"] = 20 if rsi <= 30 else 16 if rsi <= 40 else 12 if rsi <= 55 else 7 if rsi <= 70 else 3
        s["trend"] = 20 if ma5 > ma25 > ma75 else 14 if ma25 > ma75 else 10 if ma5 > ma25 else 4
        if ma25 > ma75 and -8 <= dev25 <= -1:
            s["dip"] = 20
        elif ma25 > ma75 and -1 < dev25 <= 3:
            s["dip"] = 13
        elif dev25 <= -8:
            s["dip"] = 9
        elif dev25 >= 8:
            s["dip"] = 4
        else:
            s["dip"] = 10
        if v_macd > v_signal and v_hist > hist_prev:
            s["macd"] = 20
        elif v_macd > v_signal:
            s["macd"] = 14
        elif v_macd < v_signal and v_hist > hist_prev:
            s["macd"] = 11
        else:
            s["macd"] = 5
        pos = (last_close - recent_low) / (recent_high - recent_low + 1e-9) * 100
        s["pos"] = 20 if pos <= 25 else 14 if pos <= 50 else 9 if pos <= 75 else 4
        total_score = sum(s.values())

        sig = []
        if rsi >= 75 or dev25 >= 12:
            sig.append("過熱注意")
        if rsi <= 30 and ma25 > ma75:
            sig.append("押し目候補")
        if vol_ratio >= 2:
            sig.append("出来高急増")
        signal_text = " / ".join(sig) if sig else "—"

        tech = []
        if rsi <= 30:
            tech.append(f"RSI{rsi:.0f}(売られすぎ)")
        elif rsi >= 70:
            tech.append(f"RSI{rsi:.0f}(買われすぎ)")
        else:
            tech.append(f"RSI{rsi:.0f}(中立)")
        if ma5 > ma25 > ma75:
            tech.append("移動平均=上昇配列")
        elif ma25 > ma75:
            tech.append("中期上昇")
        else:
            tech.append("中期下降")
        tech.append("MACD上向き" if v_macd > v_signal else "MACD下向き")
        tech.append(f"25日線乖離{dev25:+.1f}%")
        tech_text = " / ".join(tech)

        rows.append({
            "ティッカー": t, "銘柄": name_map.get(t, t),
            "終値$": round(last_close, 2), "前日比%": round(chg, 2),
            "RSI": round(rsi, 1), "25日線乖離%": round(dev25, 1),
            "出来高倍率": round(vol_ratio, 1), "シグナル": signal_text,
            "テクニカル要約": tech_text,
            "RSIスコア": s["RSI"], "トレンドスコア": s["trend"],
            "押し目スコア": s["dip"], "MACDスコア": s["macd"],
            "価格位置スコア": s["pos"], "総合スコア": total_score,
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def run_backtest(index_code, hold_days):
    raw, tickers, name_map, _ = load_raw(index_code)
    records = []
    for t in tickers:
        try:
            close = raw[t]["Close"].dropna()
            volume = raw[t]["Volume"].dropna()
        except Exception:
            continue
        if len(close) < 80 + hold_days:
            continue
        past_close = close.iloc[:-hold_days]
        past_volume = volume.iloc[:-hold_days]
        sc = score_from_series(past_close, past_volume)
        if sc is None:
            continue
        entry_price = float(past_close.iloc[-1])
        now_price = float(close.iloc[-1])
        ret = (now_price - entry_price) / entry_price * 100
        records.append({"ticker": t, "score": sc["total"], "return": ret})
    return pd.DataFrame(records)

# ===== UI =====
index_label = st.selectbox("対象とする市場（指数）を選択", list(INDEX_OPTIONS.keys()))
index_code = INDEX_OPTIONS[index_label]

with st.spinner(f"{index_label} を取得・計算中…（銘柄数により1〜数分）"):
    raw, tickers, name_map, fetched_at = load_raw(index_code)
    df = build_scores_full(raw, tickers, name_map)

df = df.sort_values("総合スコア", ascending=False).reset_index(drop=True)
df.index = df.index + 1

top_n = 30

c3, c4 = st.columns(2)
with c3:
    view_mode = st.selectbox(
        "目的で絞り込む",
        ["すべて表示", "買い検討（押し目候補）", "売り検討（過熱注意）", "出来高急増"]
    )
with c4:
    st.markdown("**データ最終取得**")
    st.caption(f"{fetched_at.strftime('%Y年%m月%d日 %H:%M')}（日本時間）　※株価は15〜20分遅れの遅延データ")

view = df.copy()
if view_mode == "買い検討（押し目候補）":
    view = view[view["シグナル"].str.contains("押し目候補")].sort_values("総合スコア", ascending=False)
elif view_mode == "売り検討（過熱注意）":
    view = view[view["シグナル"].str.contains("過熱注意")].sort_values("RSI", ascending=False)
elif view_mode == "出来高急増":
    view = view[view["シグナル"].str.contains("出来高急増")].sort_values("出来高倍率", ascending=False)

view = view.head(top_n)
if view.empty:
    st.warning("この条件に当てはまる銘柄は今はありません。")
    st.stop()

# ===== ランキング表 =====
st.subheader("総合スコアランキング　※30位までを表示")
st.caption(f"該当 {len(view)} 銘柄を表示中")

def color_chg(v):
    return "color: green" if v > 0 else ("color: red" if v < 0 else "")

table_cols = ["ティッカー", "銘柄", "終値$", "前日比%", "RSI", "25日線乖離%",
              "出来高倍率", "シグナル", "RSIスコア", "トレンドスコア",
              "押し目スコア", "MACDスコア", "価格位置スコア", "総合スコア"]
view_tbl = view[table_cols]

styled = (view_tbl.style
    .background_gradient(subset=["総合スコア"], cmap="RdYlGn", vmin=40, vmax=100)
    .bar(subset=["RSIスコア", "トレンドスコア", "押し目スコア", "MACDスコア", "価格位置スコア"],
         color="#9ad0ec", vmin=0, vmax=20)
    .map(color_chg, subset=["前日比%"])
    .format({"終値$": "{:.2f}", "前日比%": "{:+.2f}%", "RSI": "{:.1f}",
             "25日線乖離%": "{:+.1f}%", "出来高倍率": "{:.1f}倍"})
)

row_height = 36
table_height = min((len(view_tbl) + 1) * row_height + 3, 700)

event = st.dataframe(
    styled,
    use_container_width=True,
    height=table_height,
    on_select="rerun",
    selection_mode="single-row",
    key="rank_table",
)

# ===== 上位3銘柄 =====
st.subheader("上位3銘柄のテクニカル状態")
top3 = view.head(3).reset_index()
cols = st.columns(3)
medals = ["1位", "2位", "3位"]
for i, (_, r) in enumerate(top3.iterrows()):
    with cols[i]:
        st.metric(f"{medals[i]}　{r['ティッカー']}", f"{r['総合スコア']} 点",
                  f"前日比 {r['前日比%']:+.2f}%")
        st.progress(int(r["総合スコア"]) / 100)
        st.write(r["テクニカル要約"])
        if r["シグナル"] != "—":
            st.caption(f"検出シグナル: {r['シグナル']}")

st.divider()

# ===== 銘柄チャート =====
st.subheader("銘柄チャート（株価＋移動平均線＋RSI）")

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
ma5, ma25, ma75 = close.rolling(5).mean(), close.rolling(25).mean(), close.rolling(75).mean()
rsi = calc_rsi(close)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3], vertical_spacing=0.05,
                    subplot_titles=(f"{sel_ticker} 株価と移動平均線", "RSI"))
fig.add_trace(go.Scatter(x=close.index, y=close, name="終値", line=dict(color="black")), row=1, col=1)
fig.add_trace(go.Scatter(x=ma5.index, y=ma5, name="5日線", line=dict(color="orange")), row=1, col=1)
fig.add_trace(go.Scatter(x=ma25.index, y=ma25, name="25日線", line=dict(color="blue")), row=1, col=1)
fig.add_trace(go.Scatter(x=ma75.index, y=ma75, name="75日線", line=dict(color="green")), row=1, col=1)
fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name="RSI", line=dict(color="purple")), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h"),
                  margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===== 簡易検証機能（直感的バージョン） =====
st.subheader("スコアの簡易検証 ─ 高スコアは本当に上がった？")
st.caption("過去のある時点でスコアを計算し、そこから現在までに各銘柄がどれだけ動いたかを集計します。")

hold_label = st.selectbox("いつのスコアで検証する？",
                          ["20営業日前（約1か月）", "40営業日前（約2か月）", "60営業日前（約3か月）"])
hold_map = {"20営業日前（約1か月）": 20, "40営業日前（約2か月）": 40, "60営業日前（約3か月）": 60}
hold_days = hold_map[hold_label]

with st.spinner("過去スコアとその後の値動きを計算中…"):
    bt = run_backtest(index_code, hold_days)

if bt.empty:
    st.warning("検証に必要な期間のデータが不足しています。短い期間を選んでください。")
else:
    def band(x):
        if x >= 80:
            return "高スコア（80点以上）"
        elif x >= 60:
            return "中スコア（60〜79点）"
        else:
            return "低スコア（60点未満）"
    bt["帯"] = bt["score"].apply(band)
    order = ["高スコア（80点以上）", "中スコア（60〜79点）", "低スコア（60点未満）"]

    means = bt.groupby("帯")["return"].mean().reindex(order)
    counts = bt.groupby("帯")["return"].count().reindex(order)
    winrates = bt.assign(win=bt["return"] > 0).groupby("帯")["win"].mean().reindex(order) * 100

    high = means.get("高スコア（80点以上）")
    low = means.get("低スコア（60点未満）")

    # ---- 結論を一言で（一番上に大きく） ----
    if high is not None and low is not None:
        diff = high - low
        if diff >= 3:
            st.success(f"✅ 結論：**スコアが高い銘柄ほど、その後よく上がっていました**　"
                       f"（高スコアは低スコアより平均 {diff:+.1f}ポイント高いリターン）")
        elif diff <= -3:
            st.error(f"⚠ 結論：**この期間は、スコアが高いほど逆に伸び悩んでいました**　"
                     f"（高スコアは低スコアより平均 {diff:+.1f}ポイント低いリターン）。配点の見直し余地あり。")
        else:
            st.info("➖ 結論：**スコアの高さと値動きに、はっきりした差は出ませんでした**　"
                    "（この期間では）。相場全体の地合いに左右された可能性があります。")
    else:
        st.info("一部のスコア帯にデータが足りず、比較が十分にできませんでした。")

    # ---- スコア帯ごとのカード（色・矢印で直感的に） ----
    st.write("##### スコア帯ごとの「その後の平均リターン」")
    ccols = st.columns(3)
    palette = {"高スコア（80点以上）": "#2ca02c", "中スコア（60〜79点）": "#ff7f0e", "低スコア（60点未満）": "#d62728"}
    for i, b in enumerate(order):
        m = means.get(b)
        if m is None or pd.isna(m):
            with ccols[i]:
                st.metric(b, "データ不足")
            continue
        arrow = "📈" if m > 0 else "📉"
        with ccols[i]:
            st.metric(f"{arrow} {b}", f"{m:+.1f}%",
                      f"勝率 {winrates.get(b):.0f}%（{int(counts.get(b))}銘柄）")

    # ---- 散布図（点が右肩上がりなら関係あり） ----
    st.write("##### スコアとその後リターンの関係（点が右上がりほど関係が強い）")
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=bt["score"], y=bt["return"], mode="markers",
        marker=dict(size=8, color=bt["return"], colorscale="RdYlGn",
                    cmin=-20, cmax=20, line=dict(width=0.5, color="gray")),
        text=bt["ticker"], hovertemplate="%{text}<br>スコア %{x}<br>その後 %{y:.1f}%<extra></extra>"))
    # 傾向線（単回帰）
    if len(bt) >= 3:
        coef = pd.np.polyfit(bt["score"], bt["return"], 1) if hasattr(pd, "np") else None
    import numpy as np
    coef = np.polyfit(bt["score"], bt["return"], 1)
    xline = np.array([bt["score"].min(), bt["score"].max()])
    yline = coef[0] * xline + coef[1]
    fig_sc.add_trace(go.Scatter(x=xline, y=yline, mode="lines",
                                line=dict(color="black", dash="dash"), name="傾向線"))
    fig_sc.add_hline(y=0, line_color="gray", line_width=1)
    fig_sc.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                         xaxis_title="その時点の総合スコア", yaxis_title="その後のリターン%",
                         showlegend=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    slope = coef[0]
    if slope > 0.05:
        st.caption(f"傾向線は右上がり（傾き {slope:+.2f}）＝スコアが高いほどリターンも高め、という関係がこの期間では見られます。")
    elif slope < -0.05:
        st.caption(f"傾向線は右下がり（傾き {slope:+.2f}）＝この期間はスコアと逆の関係でした。")
    else:
        st.caption(f"傾向線はほぼ水平（傾き {slope:+.2f}）＝スコアと値動きの関係は弱めです。")

    st.caption("⚠ これは過去データでの傾向で、将来を保証するものではありません。"
               "サンプル数が少ない期間や相場全体が一方向に動いた局面では偏りが出ます。スコアの傾向確認用としてご利用ください。")

st.divider()
st.caption("構成銘柄は yfiua/index-constituents、株価は yfinance の遅延データ。スコア・シグナルは一例で上がる確率ではありません。最終判断はご自身で。")
