import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np


from datetime import datetime

# User inputs
ticker = st.text_input("Enter the ticker symbol:", "AAPL")
start_date_str = st.text_input("Enter the start date (YYYY-MM-DD):", "2010-01-01")
start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
initial_shares = st.number_input("Enter the initial number of shares:", min_value=0, value=10)
investment_frequency = st.selectbox("Select investment frequency:",
                                    ["None", "Weekly", "Bi-Weekly", "Monthly", "Yearly"])
investment_type = st.selectbox("Select investment type:", ["Money", "Shares"])
investment_amount = st.number_input("Enter the investment amount:", min_value=0, value=100)

# Download historical data
stock = yf.Ticker(ticker)
data = stock.history(start=start_date)

# Get the list of valid trading days
valid_trading_days = data.index.tolist()

# Initialize variables
shares = initial_shares
dividend_balance = 0
total_dividends = 0
all_dividends = 0
cumulative_shares = initial_shares
total_investment = initial_shares * data.iloc[0]["Close"]
last_investment_date = data.index[0]

# Create a DataFrame to store the results
results = pd.DataFrame(
    columns=["Date", "Event", "Shares", "Close Price", "Dividend Balance", "Portfolio Value", "Investment Status"])

# Add initial investment to results
results.loc[len(results)] = {"Date": data.index[0], "Event": "Initial Investment", "Shares": shares,
                             "Close Price": data.iloc[0]["Close"], "Dividend Balance": dividend_balance,
                             "Portfolio Value": shares * data.iloc[0]["Close"], "Investment Status": total_investment}

# Check if the stock pays dividends
pays_dividends = "Dividends" in data.columns

# Iterate over the data by date
for date, row in data.iterrows():
    # Handle stock splits
    if row["Stock Splits"] > 0:
        split_factor = row["Stock Splits"]
        shares *= split_factor
        cumulative_shares *= split_factor
        results.loc[len(results)] = {"Date": date, "Event": "Stock Split", "Shares": shares,
                                     "Close Price": row["Close"], "Dividend Balance": dividend_balance,
                                     "Portfolio Value": shares * row["Close"], "Investment Status": total_investment}

    # Handle dividend reinvestment (if the stock pays dividends)
    if pays_dividends and row["Dividends"] > 0:
        total_dividends = cumulative_shares * row["Dividends"]
        all_dividends+=total_dividends
        dividend_balance += total_dividends
        if dividend_balance >= row["Close"]:
            reinvested_shares = int(dividend_balance / row["Close"])
            dividend_balance -= reinvested_shares * row["Close"]
            cumulative_shares += reinvested_shares
            shares+=reinvested_shares
            results.loc[len(results)] = {"Date": date, "Event": "Dividend Reinvestment", "Shares": shares,
                                         "Close Price": row["Close"], "Dividend Balance": dividend_balance,
                                         "Portfolio Value": shares * row["Close"], "Investment Status": total_investment}

    # Handle regular investments
    if investment_frequency != "None":
        if investment_frequency == "Weekly":
            investment_date = last_investment_date + pd.DateOffset(weeks=1)
        elif investment_frequency == "Bi-Weekly":
            investment_date = last_investment_date + pd.DateOffset(weeks=2)
        elif investment_frequency == "Monthly":
            investment_date = last_investment_date + pd.DateOffset(months=1)
        elif investment_frequency == "Yearly":
            investment_date = last_investment_date + pd.DateOffset(years=1)

        while investment_date <= date:
            if investment_type == "Money":
                # Accumulate the investment amount in dividend_balance
                dividend_balance += investment_amount
                while dividend_balance >= row["Close"]:
                    shares_to_buy = int(dividend_balance / row["Close"])
                    total_investment += shares_to_buy * row["Close"]
                    shares += shares_to_buy
                    cumulative_shares += shares_to_buy
                    dividend_balance -= shares_to_buy * row["Close"]
                results.loc[len(results)] = {"Date": investment_date, "Event": f"{investment_frequency} Investment", "Shares": shares,
                                             "Close Price": row["Close"], "Dividend Balance": dividend_balance,
                                             "Portfolio Value": shares * row["Close"], "Investment Status": total_investment}
            else:  # investment_type == "Shares"
                if investment_amount > 0:
                    total_investment += investment_amount * row["Close"]
                    shares += investment_amount
                    cumulative_shares += investment_amount
                    results.loc[len(results)] = {"Date": investment_date, "Event": f"{investment_frequency} Investment", "Shares": shares,
                                                 "Close Price": row["Close"], "Dividend Balance": dividend_balance,
                                                 "Portfolio Value": shares * row["Close"], "Investment Status": total_investment}
            last_investment_date = investment_date

            if investment_frequency == "Weekly":
                investment_date += pd.DateOffset(weeks=1)
            elif investment_frequency == "Bi-Weekly":
                investment_date += pd.DateOffset(weeks=2)
            elif investment_frequency == "Monthly":
                investment_date += pd.DateOffset(months=1)
            elif investment_frequency == "Yearly":
                investment_date += pd.DateOffset(years=1)

# Calculate portfolio value
portfolio_value = cumulative_shares * data.iloc[-1]["Close"]
results.loc[len(results)] = {"Date": data.index[-1], "Event": "Current Value", "Shares": shares,
                             "Close Price": data.iloc[-1]["Close"], "Dividend Balance": dividend_balance,
                             "Portfolio Value": portfolio_value, "Investment Status": total_investment}

# Calculate portfolio progress summary
total_return = portfolio_value - total_investment
annualized_return = (np.power(portfolio_value / total_investment, 1 / len(data.index.year.unique())) - 1) * 100

# Display the results in Streamlit
st.markdown("## Investment Results")
st.write(results)

st.markdown("## Summary")
st.write(f"Total investment: ${total_investment:.2f}")
st.write(f"Total dividends: ${all_dividends:.2f}")
st.write(f"Total return: ${total_return:.2f}")
st.write(f"Annualized return: {annualized_return:.2f}%")

# Create a plotly figure for the portfolio value and total investment over time
fig = go.Figure()
fig.add_trace(go.Scatter(x=results["Date"], y=results["Portfolio Value"], mode='lines+markers', name='Portfolio Value'))

fig.update_layout(title='Portfolio Value and Total Investment Over Time', xaxis_title='Date', yaxis_title='Value ($)',
                  hovermode="x")

# Create a pie chart for the portfolio composition
composition_labels = ['Investment', 'Dividends', 'Gain']
composition_values = [total_investment, total_dividends, total_return]

fig_pie = go.Figure(data=[go.Pie(labels=composition_labels, values=composition_values, hole=0.3)])
fig_pie.update_layout(title_text="Portfolio Composition")

# Display the figures
st.plotly_chart(fig)
st.plotly_chart(fig_pie)
