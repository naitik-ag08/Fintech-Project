import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
## Loading the data
ticker = "infy"
start_date = "2025-01-01"
end_date = "2026-01-01"

df = yf.download(ticker, start=start_date, end=end_date)
df.columns = df.columns.get_level_values(0)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)


close = df['Close'].tolist()

st.set_page_config(
    page_title='Infosys - Python Project',
    page_icon='📊',
    layout='centered'
)


## Indicators 

def sma(series, window):
    sma = []
    for i in range(len(series)):
        if i < window -1:
            sma.append(np.nan)
        else:
            sma.append(np.mean(series[i - window + 1 : i + 1]))

    return sma

def ema(series, window):
    k = 2 / (window + 1)
    ema_vals = [series[0]]
    for i in range(1, len(series)):
      ema_vals.append(series[i]*k + ema_vals[i-1]*(1-k))
    return ema_vals

def rsi(series, period=14):
    gains, losses, = [], []
    for i in range(1, len(series)):
        diff = series[i] - series[i-1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

    rsi_vals = [None]*period
    for i in range(period, len(series)):
        avg_gain = sum(gains[i-period:i]) / period 
        avg_loss = sum(losses[i-period:i]) / period
        if avg_loss == 0:
            rsi_vals.append(100)
        else:
            rs = (avg_gain) / (avg_loss)
            rsi_vals.append(100 - (100/(1+rs)))
    return rsi_vals

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = [ema12[i] - ema26[i] 
  for i in range(len(series))]
    signal = ema(macd_line, 9)
    return macd_line, signal

def bollinger(series, window=20):
    upper, lower = [], []
    for i in range(len(series)):
        if i < window-1:
            upper.append(np.nan)
            lower.append(np.nan)
        else:
            data = series[i-window+1:i+1]
            mean = sum(data) / window
            std = np.std(data)
            upper.append(mean + 2*std)
            lower.append(mean - 2*std)
    return upper, lower

df['SMA_20'] = sma(close, 20)
df['EMA_20'] = ema(close, 20)
df['RSI_14'] = rsi(close)
df['MACD'], df['MACD_Signal'] = macd(close)
df['BB_Upper'], df['BB_Lower'] = bollinger(close)


## Final Indicator values

signals = []

for i in range(len(df)):
    buy = 0
    sell = 0

    if df['Close'][i] > df['SMA_20'][i]: buy += 1
    if df['Close'][i] < df['SMA_20'][i]: sell -= 1

    if df['Close'][i] > df['EMA_20'][i]: buy += 1
    if df['Close'][i] < df['EMA_20'][i]: sell -= 1

    if df['RSI_14'][i] is not None:
        if df['RSI_14'][i] < 30: buy += 1
        if df['RSI_14'][i] > 70: sell -= 1

    if df['MACD'][i] > df['MACD_Signal'][i]: buy += 1
    if df['MACD'][i] < df['MACD_Signal'][i]: sell -= 1

    if df['BB_Lower'][i] is not None:
        if df['Close'][i] < df['BB_Lower'][i]: buy += 1
        if df['Close'][i] > df['BB_Upper'][i]: sell -= 1

    if buy >= 2:
        signals.append(1)
    elif sell <= -2:
        signals.append(-1)
    else:
        signals.append(0)

final_signal= []
prev_state = 0

for i in range(len(signals)):
    if signals[i] == 1 and prev_state != 1:
        final_signal.append(1) #BUY
        prev_state = 1
    elif signals[i] != 1 and prev_state == 1:
        final_signal.append(-1) #SELL
        prev_state = 0
    else:
        final_signal.append(0) #HOLD

df['Final_Signal'] = final_signal


st.title('TradeBook Section')
with st.expander('View TradeBook 📖'):
    st.dataframe(df)


st.title('Graphs of Indicators')

st.subheader('Close vs EMA20 and SMA20')

fig_ma = go.Figure()

fig_ma.add_trace(go.Scatter(
    y=df['Close'],
    mode='lines', name='Close Price'
))
fig_ma.add_trace(go.Scatter(
    y=df['SMA_20'],
    mode='lines', name='SMA 20'
))
fig_ma.add_trace(go.Scatter(
    y=df['EMA_20'],
    mode='lines', name='EMA 20'
))

fig_ma.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=450
)

st.plotly_chart(fig_ma, use_container_width=True)



st.subheader("📉 RSI Indicator")

fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
    y=df['RSI_14'],
    mode='lines', name='RSI'
))

fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="Oversold (30)")
fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="Overbought (70)")

fig_rsi.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    height=350
)

st.plotly_chart(fig_rsi, use_container_width=True)

st.subheader("📊 Bollinger Bands")

fig_bb = go.Figure()

fig_bb.add_trace(go.Scatter(
    y=df['Close'],
    mode='lines', name='Close Price'
))
fig_bb.add_trace(go.Scatter(
    y=df['BB_Upper'],
    mode='lines', name='Upper Band'
))
fig_bb.add_trace(go.Scatter(
    y=df['BB_Lower'],
    mode='lines', name='Lower Band'
))

fig_bb.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=450
)

st.plotly_chart(fig_bb, use_container_width=True)

st.subheader("📈 Strategy vs Market Performance")
df['Returns'] = df['Close'].pct_change()
df['Strategy_Returns'] = df['Returns'] * df['Final_Signal'].shift(1)
market = (1 + df['Returns']).cumprod()
strategy = (1 + df['Strategy_Returns']).cumprod()

fig_perf = go.Figure()

fig_perf.add_trace(go.Scatter(
    y=market,
    mode='lines', name='Market Returns'
))
fig_perf.add_trace(go.Scatter(
    y=strategy,
    mode='lines', name='Strategy Returns'
))

fig_perf.update_layout(
    xaxis_title="Date",
    yaxis_title="Cumulative Returns",
    height=400
)

st.plotly_chart(fig_perf, use_container_width=True)

st.subheader("🟢 Buy / 🔴 Sell Signals")
buy_signals = df[df['Final_Signal'] == 1]
sell_signals = df[df['Final_Signal'] == -1]

fig_signal = go.Figure()

fig_signal.add_trace(go.Scatter(
    x=df.index, y=df['Close'],
    mode='lines', name='Close Price'
))

fig_signal.add_trace(go.Scatter(
    x=buy_signals.index,
    y=buy_signals['Close'],
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Buy'
))

fig_signal.add_trace(go.Scatter(
    x=sell_signals.index,
    y=sell_signals['Close'],
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Sell'
))

fig_signal.update_layout(
    xaxis_title="Date",
    yaxis_title="Price",
    height=450
)

st.plotly_chart(fig_signal, use_container_width=True)

