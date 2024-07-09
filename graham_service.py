# import streamlit as st
# import yfinance as yf
# import math
# import mplfinance as mpf
# import pandas as pd
# import numpy as np
# from scipy.signal import argrelextrema
#
# def calculate_graham_number(eps, book_value):
#     if eps < 0 or book_value < 0:
#         return None
#     return round(math.sqrt(22.5 * eps * book_value), 2)
#
# def calculate_dcf(free_cash_flow, discount_rate):
#     # Assuming a simple DCF model with a constant growth rate
#     growth_rate = 0.05  # Adjust as needed
#     terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
#     dcf = terminal_value / (1 + discount_rate) ** 5  # Assume 5 years for simplicity
#     return round(dcf, 2)
#
# def analyze_trend(ticker):
#     try:
#         data = yf.download(ticker)
#         stock_info = yf.Ticker(ticker).info
#         company_name = stock_info['longName']
#         trailing_eps = stock_info['trailingEps']
#         book_value = stock_info['bookValue']
#         free_cash_flow = stock_info.get('freeCashflow', 0)
#
#         graham_num = calculate_graham_number(trailing_eps, book_value)
#         dcf = calculate_dcf(free_cash_flow, discount_rate=0.1)  # Use an appropriate discount rate
#
#         if graham_num is not None:
#             difference_graham = round(graham_num - stock_info['currentPrice'], 2)
#         else:
#             difference_graham = 'N/A'
#
#         if dcf is not None:
#             difference_dcf = round(dcf - stock_info['currentPrice'], 2)
#         else:
#             difference_dcf = 'N/A'
#
#         return {'name': company_name, 'current_price': stock_info['currentPrice'], 'graham_number': graham_num,
#                 'difference_graham': difference_graham, 'dcf': dcf, 'difference_dcf': difference_dcf,
#                 'description': stock_info.get('longBusinessSummary', 'N/A'),
#                 'sector': stock_info.get('sector', 'N/A'), 'data': data}
#     except Exception as e:
#         st.error(f"An error occurred while analyzing {ticker}: {e}")
#
# def plot_fibonacci_retracements(data):
#     # Filter data for a maximum of 5 years
#     data = data[-252 * 5:]  # Assuming 252 trading days in a year
#
#     # Calculate Fibonacci levels
#     high = data['High'].max()
#     low = data['Low'].min()
#     diff = high - low
#     retracement_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236]
#     retracements = [high - level * diff for level in retracement_levels]
#
#     # Identify local maxima and minima for support and resistance lines
#     data['min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal)[0]]['Low']
#     data['max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal)[0]]['High']
#
#     # Create plot with type='candle'
#     fig, axes = mpf.plot(data, type='candle', returnfig=True, figsize=(12, 8),
#                          title=f"Stock Price Analysis for {data.index[0].date()} to {data.index[-1].date()}")
#     ax = axes[0]
#
#     # Plot Fibonacci retracement levels
#     for i, level in enumerate(retracements):
#         ax.axhline(level, linestyle='--', alpha=0.7, linewidth=2.0, color=f'C{i}', label=f'Fibonacci Level {retracement_levels[i]}')
#
#     # Plot the next resistance and previous support for the current price
#     current_price = result['current_price']
#     next_resistance = data['max'].loc[data['max'] > current_price].min()
#     previous_support = data['min'].loc[data['min'] < current_price].max()
#
#     if not np.isnan(next_resistance):
#         ax.axhline(next_resistance, linestyle='-', color='orange', linewidth=2.0, label='Next Resistance')
#
#     if not np.isnan(previous_support):
#         ax.axhline(previous_support, linestyle='-', color='purple', linewidth=2.0, label='Previous Support')
#
#     # Highlight current price
#     ax.axhline(result['current_price'], linestyle='-', color='blue', linewidth=2.0, label='Current Price')
#
#     # Remove unnecessary elements
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles=handles[1:], labels=labels[1:])  # Create a legend excluding the first item
#     ax.set_xticks([])  # Remove x-axis ticks
#     ax.set_xticklabels([])  # Remove x-axis tick labels
#
#     # Enhancements
#     ax.set_ylim([data['Low'].min() * 0.95, data['High'].max() * 1.05])  # Adjust y-axis range for better visibility
#     ax.set_ylabel('Stock Price')
#     ax.set_xlabel('Date')
#
#     # Add annotations for interesting zones
#     for i, level in enumerate(retracements[:-1]):
#         ax.annotate(f'Fib {retracement_levels[i]} - {retracement_levels[i + 1]}', xy=(data.index[0], level),
#                     xytext=(data.index[0], level * 0.95),
#                     arrowprops=dict(facecolor=f'C{i}', arrowstyle='->'),
#                     fontsize=10, color=f'C{i}')
#
#     st.write('Fibonacci Levels:')
#     fibonacci_table = pd.DataFrame({
#         'Level': retracement_levels,
#         'Price': retracements
#     })
#     st.table(fibonacci_table)
#
#     return fig
#
# def plot_moving_averages(data):
#     # Calculate moving averages
#     data['SMA50'] = data['Close'].rolling(window=50).mean()
#     data['SMA200'] = data['Close'].rolling(window=200).mean()
#
#     # Plotting moving averages
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(data.index, data['Close'], label='Close Price')
#     ax.plot(data.index, data['SMA50'], label='50-day SMA', linestyle='--')
#     ax.plot(data.index, data['SMA200'], label='200-day SMA', linestyle='--')
#
#     # Plotting buy and sell signals based on moving average crossover
#     buy_signals = data[(data['SMA50'] > data['SMA200']) & (data['SMA50'].shift(1) < data['SMA200'].shift(1))]
#     sell_signals = data[(data['SMA50'] < data['SMA200']) & (data['SMA50'].shift(1) > data['SMA200'].shift(1))]
#     ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')
#     ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal')
#
#     ax.set_title('Moving Averages Crossover Strategy')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     ax.legend()
#
#     return fig
#
# st.title('Stock Analysis with Graham Number, DCF, and Technical Analysis')
#
# ticker = st.text_input('Enter a ticker symbol:', '')
#
# if ticker:
#     result = analyze_trend(ticker)
#     if result:
#         st.subheader(f"**{result['name']}**")
#         st.write(f"**Current Price**: {result['current_price']}")
#         if result['graham_number'] is None:
#             st.warning("The Graham Number couldn't be calculated because the EPS or Book Value is negative.")
#         else:
#             st.write(f"**Graham Number**: {result['graham_number']}")
#             st.write(f"**Difference (Graham)**: {result['difference_graham']}")
#             st.write(f"**DCF Value**: {result['dcf']}")
#             st.write(f"**Difference (DCF)**: {result['difference_dcf']}")
#         st.subheader('Company Overview')
#         st.write(f"**Description**: {result['description']}")
#         st.write(f"**Sector**: {result['sector']}")
#         st.subheader('Fibonacci Retracements and Support/Resistance Lines')
#         fig = plot_fibonacci_retracements(result['data'])
#         st.pyplot(fig)
#         st.subheader('Moving Averages Crossover Strategy')
#         fig_ma = plot_moving_averages(result['data'])
#         st.pyplot(fig_ma)
#     else:
#         st.warning("Couldn't calculate the Graham Number for this ticker.")
# import streamlit as st
# import yfinance as yf
# import math
# import mplfinance as mpf
# import pandas as pd
# import numpy as np
# from scipy.signal import argrelextrema
# import matplotlib.pyplot as plt
#
#
#
# def calculate_graham_number(eps, book_value):
#     if eps < 0 or book_value < 0:
#         return None
#     return round(math.sqrt(22.5 * eps * book_value), 2)
#
#
# def calculate_dcf(free_cash_flow, discount_rate):
#     growth_rate = 0.05  # Adjust as needed
#     terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
#     dcf = terminal_value / (1 + discount_rate) ** 5  # Assume 5 years for simplicity
#     return round(dcf, 2)
#
#
# def analyze_trend(ticker):
#     try:
#         data = yf.download(ticker)
#         stock_info = yf.Ticker(ticker).info
#         company_name = stock_info['longName']
#         trailing_eps = stock_info['trailingEps']
#         book_value = stock_info['bookValue']
#         free_cash_flow = stock_info.get('freeCashflow', 0)
#         current_price = stock_info['currentPrice']
#
#         graham_num = calculate_graham_number(trailing_eps, book_value)
#         dcf = calculate_dcf(free_cash_flow, discount_rate=0.1)  # Use an appropriate discount rate
#
#         if graham_num is not None:
#             difference_graham = round(graham_num - current_price, 2)
#         else:
#             difference_graham = 'N/A'
#
#         if dcf is not None:
#             difference_dcf = round(dcf - current_price, 2)
#         else:
#             difference_dcf = 'N/A'
#
#         return {
#             'name': company_name,
#             'current_price': current_price,
#             'graham_number': graham_num,
#             'difference_graham': difference_graham,
#             'dcf': dcf,
#             'difference_dcf': difference_dcf,
#             'description': stock_info.get('longBusinessSummary', 'N/A'),
#             'sector': stock_info.get('sector', 'N/A'),
#             'data': data
#         }
#     except Exception as e:
#         st.error(f"An error occurred while analyzing {ticker}: {e}")
#
#
# def plot_fibonacci_retracements(data, current_price):
#     data = data[-252 * 5:]  # Last 5 years of data
#     high = data['High'].max()
#     low = data['Low'].min()
#     diff = high - low
#     retracement_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236]
#     retracements = [high - level * diff for level in retracement_levels]
#
#     data['min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=5)[0]]['Low']
#     data['max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=5)[0]]['High']
#
#     fig, ax = plt.subplots(figsize=(12, 8))
#     #print(data)
#
#     mpf.plot(data, type='candle', ax=ax, title=f"Stock Price Analysis")
#
#     colors = ['#FFEBEE', '#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#F44336']
#     for i in range(len(retracement_levels) - 1):
#         ax.fill_between(data.index, retracements[i], retracements[i + 1], color=colors[i], alpha=0.2,
#                         label=f'Fib {retracement_levels[i]} - {retracement_levels[i + 1]}')
#
#     next_resistance = data['max'].loc[data['max'] > current_price].min()
#     previous_support = data['min'].loc[data['min'] < current_price].max()
#
#     if not np.isnan(next_resistance):
#         ax.axhline(next_resistance, linestyle='-', color='orange', linewidth=2.0, label='Next Resistance')
#
#     if not np.isnan(previous_support):
#         ax.axhline(previous_support, linestyle='-', color='purple', linewidth=2.0, label='Previous Support')
#
#     ax.axhline(current_price, linestyle='-', color='blue', linewidth=2.0, label='Current Price')
#
#     ax.legend()
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim([data['Low'].min() * 0.95, data['High'].max() * 1.05])
#     ax.set_ylabel('Stock Price')
#     ax.set_xlabel('Date')
#
#     st.write('Fibonacci Levels:')
#     fibonacci_table = pd.DataFrame({'Level': retracement_levels, 'Price': retracements})
#     st.table(fibonacci_table)
#
#     return fig
#
#
# def plot_moving_averages(data):
#     data['SMA50'] = data['Close'].rolling(window=50).mean()
#     data['SMA200'] = data['Close'].rolling(window=200).mean()
#
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.plot(data.index, data['Close'], label='Close Price')
#     ax.plot(data.index, data['SMA50'], label='50-day SMA', linestyle='--')
#     ax.plot(data.index, data['SMA200'], label='200-day SMA', linestyle='--')
#
#     buy_signals = data[(data['SMA50'] > data['SMA200']) & (data['SMA50'].shift(1) < data['SMA200'].shift(1))]
#     sell_signals = data[(data['SMA50'] < data['SMA200']) & (data['SMA50'].shift(1) > data['SMA200'].shift(1))]
#     ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')
#     ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal')
#
#     ax.set_title('Moving Averages Crossover Strategy')
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price')
#     ax.legend()
#
#     return fig
#
#
# def plot_rsi_macd(data):
#     # Calculate RSI
#     delta = data['Close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     rsi = 100 - (100 / (1 + rs))
#     data['RSI'] = rsi
#
#     # Calculate MACD
#     exp1 = data['Close'].ewm(span=12, adjust=False).mean()
#     exp2 = data['Close'].ewm(span=26, adjust=False).mean()
#     macd = exp1 - exp2
#     signal = macd.ewm(span=9, adjust=False).mean()
#     data['MACD'] = macd
#     data['Signal_Line'] = signal
#
#     fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)
#
#     ax[0].plot(data.index, data['RSI'], label='RSI', color='blue')
#     ax[0].axhline(70, linestyle='--', alpha=0.5, color='red')
#     ax[0].axhline(30, linestyle='--', alpha=0.5, color='green')
#     ax[0].set_title('Relative Strength Index (RSI)')
#     ax[0].legend()
#
#     ax[1].plot(data.index, data['MACD'], label='MACD', color='blue')
#     ax[1].plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
#     ax[1].bar(data.index, data['MACD'] - data['Signal_Line'], label='MACD Histogram', color='grey')
#     ax[1].set_title('Moving Average Convergence Divergence (MACD)')
#     ax[1].legend()
#
#     return fig
#
#
# st.title('Stock Analysis with Graham Number, DCF, and Technical Analysis')
#
# ticker = st.text_input('Enter a ticker symbol:', '')
#
# if ticker:
#     result = analyze_trend(ticker)
#     if result:
#         st.subheader(result['name'])
#         st.write(f"**Current Price:** ${result['current_price']}")
#         st.write(f"**Graham Number:** {result['graham_number']}")
#         st.write(f"**Difference (Graham - Current Price):** {result['difference_graham']}")
#         st.write(f"**DCF Valuation:** {result['dcf']}")
#         st.write(f"**Difference (DCF - Current Price):** {result['difference_dcf']}")
#         st.write(f"**Sector:** {result['sector']}")
#         st.write(f"**Description:** {result['description']}")
#
#         fig1 = plot_fibonacci_retracements(result['data'], result['current_price'])
#         st.pyplot(fig1)
#
#         fig2 = plot_moving_averages(result['data'])
#         st.pyplot(fig2)
#
#         fig3 = plot_rsi_macd(result['data'])
#         st.pyplot(fig3)
#     else:
#         st.error('Failed to retrieve or analyze data for the given ticker.')


import streamlit as st
import yfinance as yf
import math
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

def calculate_graham_number(eps, book_value):
    if eps < 0 or book_value < 0:
        return None
    return round(math.sqrt(22.5 * eps * book_value), 2)

def calculate_dcf(free_cash_flow, discount_rate):
    growth_rate = 0.05  # Adjust as needed
    terminal_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
    dcf = terminal_value / (1 + discount_rate) ** 5  # Assume 5 years for simplicity
    return round(dcf, 2)

def analyze_trend(ticker):
    try:
        data = yf.download(ticker, period="2y")
        stock_info = yf.Ticker(ticker).info
        company_name = stock_info['longName']
        trailing_eps = stock_info['trailingEps']
        book_value = stock_info['bookValue']
        free_cash_flow = stock_info.get('freeCashflow', 0)
        current_price = stock_info['currentPrice']

        graham_num = calculate_graham_number(trailing_eps, book_value)
        dcf = calculate_dcf(free_cash_flow, discount_rate=0.1)  # Use an appropriate discount rate

        if graham_num is not None:
            difference_graham = round(graham_num - current_price, 2)
        else:
            difference_graham = 'N/A'

        if dcf is not None:
            difference_dcf = round(dcf - current_price, 2)
        else:
            difference_dcf = 'N/A'

        return {
            'name': company_name,
            'current_price': current_price,
            'graham_number': graham_num,
            'difference_graham': difference_graham,
            'dcf': dcf,
            'difference_dcf': difference_dcf,
            'description': stock_info.get('longBusinessSummary', 'N/A'),
            'sector': stock_info.get('sector', 'N/A'),
            'data': data
        }
    except Exception as e:
        st.error(f"An error occurred while analyzing {ticker}: {e}")

# def plot_fibonacci_retracements(data, current_price):
#     data = data[-252 * 5:]  # Last 5 years of data
#     data.index = pd.to_datetime(data.index)  # Ensure the index is DateTime
#
#     high = data['High'].max()
#     low = data['Low'].min()
#     diff = high - low
#     retracement_levels = [1.0, 0.786, 0.618, 0.5, 0.382, 0.236]
#     retracements = [high - level * diff for level in retracement_levels]
#
#     data['min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=5)[0]]['Low']
#     data['max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=5)[0]]['High']
#
#     # Ensure data is in the correct format
#     data_mpf = data[['Open', 'High', 'Low', 'Close', 'Volume']]
#
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # Define a custom style without title to prevent NoneType error
#     style = mpf.make_mpf_style(base_mpf_style='charles', rc={'figure.figsize':(12,8), 'axes.labelsize':12, 'xtick.labelsize':10, 'ytick.labelsize':10})
#
#     mpf.plot(data_mpf, type='candle', ax=ax, style=style)
#
#     colors = ['#FFEBEE', '#FFCDD2', '#EF9A9A', '#E57373', '#EF5350', '#F44336']
#     for i in range(len(retracement_levels) - 1):
#         ax.fill_between(data.index, retracements[i], retracements[i + 1], color=colors[i], alpha=0.2,
#                         label=f'Fib {retracement_levels[i]} - {retracement_levels[i + 1]}')
#
#     next_resistance = data['max'].loc[data['max'] > current_price].min()
#     previous_support = data['min'].loc[data['min'] < current_price].max()
#
#     if not np.isnan(next_resistance):
#         ax.axhline(next_resistance, linestyle='-', color='orange', linewidth=2.0, label='Next Resistance')
#
#     if not np.isnan(previous_support):
#         ax.axhline(previous_support, linestyle='-', color='purple', linewidth=2.0, label='Previous Support')
#
#     ax.axhline(current_price, linestyle='-', color='blue', linewidth=2.0, label='Current Price')
#
#     ax.legend()
#     ax.set_xticks([])
#     ax.set_xticklabels([])
#     ax.set_ylim([data['Low'].min() * 0.95, data['High'].max() * 1.05])
#     ax.set_ylabel('Stock Price')
#     ax.set_xlabel('Date')
#
#     st.write('Fibonacci Levels:')
#     fibonacci_table = pd.DataFrame({'Level': retracement_levels, 'Price': retracements})
#
#     # Color coding based on preferred buying or selling regions
#     preferred_buy_levels = [0.382, 0.5, 0.618]
#
#     # Create a function to determine row color based on Fibonacci level
#     def get_row_color(level):
#         if level in preferred_buy_levels:
#             return 'background-color: #C8E6C9'  # Light green for preferred buy levels
#         else:
#             return 'background-color: #FFCDD2'  # Light red for other levels
#
#     # Apply row color formatting to the Fibonacci table
#     fibonacci_table_styled = fibonacci_table.style.applymap(lambda x: get_row_color(x), subset=['Level'])
#
#     st.table(fibonacci_table_styled)
#
#     return fig

def plot_support_resistance(data, current_price):
    data = data[-252 * 5:]  # Last 5 years of data
    data.index = pd.to_datetime(data.index)  # Ensure the index is DateTime

    high = data['High'].max()
    low = data['Low'].min()

    data['min'] = data.iloc[argrelextrema(data['Low'].values, np.less_equal, order=5)[0]]['Low']
    data['max'] = data.iloc[argrelextrema(data['High'].values, np.greater_equal, order=5)[0]]['High']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot candlestick chart
    mpf.plot(data, type='candle', ax=ax, style='charles')

    # Plot support and resistance lines
    ax.axhline(data['max'].loc[data['max'] > current_price].min(), linestyle='-', color='orange', linewidth=2.0, label='Next Resistance')
    ax.axhline(data['min'].loc[data['min'] < current_price].max(), linestyle='-', color='purple', linewidth=2.0, label='Previous Support')
    ax.axhline(current_price, linestyle='-', color='blue', linewidth=2.0, label='Current Price')

    ax.legend()
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_ylim([data['Low'].min() * 0.95, data['High'].max() * 1.05])
    ax.set_ylabel('Stock Price')
    ax.set_xlabel('Date')

    return fig


def plot_moving_averages(data):
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['SMA200'] = data['Close'].rolling(window=200).mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['SMA50'], label='50-day SMA', linestyle='--')
    ax.plot(data.index, data['SMA200'], label='200-day SMA', linestyle='--')

    buy_signals = data[(data['SMA50'] > data['SMA200']) & (data['SMA50'].shift(1) < data['SMA200'].shift(1))]
    sell_signals = data[(data['SMA50'] < data['SMA200']) & (data['SMA50'].shift(1) > data['SMA200'].shift(1))]
    ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')
    ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal')

    ax.set_title('Moving Averages Crossover Strategy')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()

    return fig

def plot_rsi_macd(data):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    data['RSI'] = rsi

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD'] = macd
    data['Signal_Line'] = signal

    fig, ax = plt.subplots(2, figsize=(12, 8), sharex=True)

    ax[0].plot(data.index, data['RSI'], label='RSI', color='blue')
    ax[0].axhline(70, linestyle='--', alpha=0.5, color='red')
    ax[0].axhline(30, linestyle='--', alpha=0.5, color='green')
    ax[0].set_title('Relative Strength Index (RSI)')
    ax[0].legend()

    ax[1].plot(data.index, data['MACD'], label='MACD', color='blue')
    ax[1].plot(data.index, data['Signal_Line'], label='Signal Line', color='red')
    ax[1].bar(data.index, data['MACD'] - data['Signal_Line'], label='MACD Histogram', color='grey')
    ax[1].set_title('Moving Average Convergence Divergence (MACD)')
    ax[1].legend()

    return fig

def generate_summary(result):
    summary = ""

    # Graham Number analysis
    if result['graham_number'] is not None:
        summary += f"**Graham Number:** {result['graham_number']}\n"
        if result['difference_graham'] < 0:
            summary += f"The current price is below the Graham Number by ${abs(result['difference_graham'])}. Consider buying.\n"
        else:
            summary += f"The current price is above the Graham Number by ${result['difference_graham']}. Consider selling.\n"
    else:
        summary += "Graham Number not available.\n"

    # DCF analysis
    if result['dcf'] is not None:
        summary += f"**Discounted Cash Flow (DCF):** {result['dcf']}\n"
        if result['difference_dcf'] < 0:
            summary += f"The current price is below the DCF value by ${abs(result['difference_dcf'])}. Consider buying.\n"
        else:
            summary += f"The current price is above the DCF value by ${result['difference_dcf']}. Consider selling.\n"
    else:
        summary += "DCF value not available.\n"

    return summary

st.title('Stock Analysis with Graham Number, DCF, and Technical Indicators')

ticker = st.text_input('Enter a ticker symbol:', '')

if ticker:
    result = analyze_trend(ticker)
    if result:
        st.subheader(result['name'])
        st.write(f"**Current Price:** ${result['current_price']}")
        st.write(f"**Sector:** {result['sector']}")
        st.write(f"**Description:** {result['description']}")

        fig1 = plot_support_resistance(result['data'], result['current_price'])
        st.pyplot(fig1)

        fig2 = plot_moving_averages(result['data'])
        st.pyplot(fig2)

        fig3 = plot_rsi_macd(result['data'])
        st.pyplot(fig3)

        # summary = generate_summary(result)
        # st.subheader('Summary')
        # st.write(summary)




