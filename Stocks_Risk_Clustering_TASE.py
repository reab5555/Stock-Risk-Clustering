import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Function to fetch data from Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    return yf.download(symbol, start=start_date, end=end_date)

# Calculate the beta of each stock against the market index
def calculate_beta(stock_returns, market_returns):
    if len(stock_returns) < 2 or len(market_returns) < 2:
        return np.nan, np.nan, np.nan
    covariance_matrix = np.cov(stock_returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    return beta, covariance_matrix[0, 1], covariance_matrix[1, 1]  # Also return covariance and variance for debugging

# Calculate the R-squared value between two stock series
def calculate_r_squared(stock_returns, market_returns):
    if len(stock_returns) < 2 or len(market_returns) < 2:
        return np.nan
    correlation_matrix = np.corrcoef(stock_returns, market_returns)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    return r_squared

# Determine the optimal number of clusters using BIC
def determine_optimal_clusters(data):
    bics = []
    n_clusters_range = range(5, 16)  # from 5 to 15 clusters

    for n_clusters in n_clusters_range:
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(data)
        bic = gmm.bic(data)
        bics.append((n_clusters, bic))

    optimal_clusters = min(bics, key=lambda x: x[1])[0]

    # Plot BIC scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, [bic for _, bic in bics], marker='o')
    plt.title('BIC Scores for Different Numbers of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('BIC Score')
    plt.show()

    print(f"The optimal number of clusters determined by BIC is: {optimal_clusters}")
    return optimal_clusters

# Align stock and market data
def align_data(stock_data, index_data):
    stock_data.index = stock_data.index.tz_localize(None)
    index_data.index = index_data.index.tz_localize(None)
    aligned_data = stock_data.join(index_data, how='inner', lsuffix='_stock', rsuffix='_index')
    return aligned_data

# Function to fetch data for Israeli stocks from Yahoo Finance
def fetch_israeli_stocks(stock_list, start_date, end_date):
    all_stocks_data = pd.DataFrame()

    for symbol in tqdm(stock_list, desc="Fetching stock data"):
        try:
            stock = yf.Ticker(symbol)
            stock_info = stock.history(start=start_date, end=end_date)
            if stock_info.empty:
                print(f"No data found for {symbol} in the given date range.")
                continue

            # Correct the timezone issue with 'dst_error_hours'
            if 'dst_error_hours' in stock_info.columns:
                stock_info.index += pd.to_timedelta(stock_info['dst_error_hours'], unit='h')

            stock_info.reset_index(inplace=True)
            stock_info['Symbol'] = symbol.replace('.TA', '')
            all_stocks_data = pd.concat([all_stocks_data, stock_info])
        except Exception as e:
            print(f"Failed to fetch data for {symbol}: {e}")

    return all_stocks_data

# Main function to process the data
def analyze_israeli_stocks(stocks_filepath, index_symbol, years=5):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    # Load the list of Tel Aviv stock symbols from the CSV file
    stock_symbols_df = pd.read_csv(stocks_filepath)
    stock_symbols = [symbol + '.TA' for symbol in stock_symbols_df['Symbol']]

    # Fetch market index data
    print(f"Fetching market index data for {index_symbol} from {start_date} to {end_date}")
    index_data = fetch_data(index_symbol, start_date, end_date)
    index_data = index_data['Close'].to_frame(name='Close_index')

    betas = {}
    r_squared_values = {}
    latest_close_values = {}
    symbols = {}
    valid_stocks_count = 0

    # Fetch stock data
    all_stocks_data = fetch_israeli_stocks(stock_symbols, start_date, end_date)

    for symbol in tqdm(stock_symbols, desc="Calculating metrics"):
        stock_data = all_stocks_data[all_stocks_data['Symbol'] == symbol.replace('.TA', '')][['Date', 'Close']]
        stock_data.set_index('Date', inplace=True)

        if not stock_data.empty:
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = index_data['Close_index'].pct_change().dropna()

            aligned_data = align_data(stock_returns.to_frame(name='Close_stock'), market_returns.to_frame(name='Close_index'))

            if not aligned_data['Close_stock'].empty and not aligned_data['Close_index'].empty:
                beta, cov, var = calculate_beta(aligned_data['Close_stock'].dropna(), aligned_data['Close_index'].dropna())
                print(f"Stock: {symbol}, Beta: {beta}, Covariance: {cov}, Variance: {var}, "
                      f"Stock Returns: {aligned_data['Close_stock'].mean()}, Market Returns: {aligned_data['Close_index'].mean()}")  # Debugging output
                if np.isfinite(beta):
                    betas[symbol] = beta
                    r_squared_values[symbol] = calculate_r_squared(aligned_data['Close_stock'].dropna(), aligned_data['Close_index'].dropna())
                    latest_close_values[symbol] = stock_data['Close'].iloc[-1]
                    symbols[symbol] = symbol
                    valid_stocks_count += 1
                else:
                    print(f"Skipping {symbol} due to non-finite beta value")
            else:
                print(f"Skipping {symbol} due to empty aligned returns")

    results_df = pd.DataFrame({
        'Symbol': list(betas.keys()),
        'Name': [stock_symbols_df[stock_symbols_df['Symbol'] == symbol.replace('.TA', '')]['Name'].values[0] for symbol in betas.keys()],
        'Beta': list(betas.values()),
        'R-Squared': [r_squared_values[symbol] for symbol in betas.keys()],
        'Latest Close': [latest_close_values[symbol] for symbol in betas.keys()]
    }).sort_values(by='Beta')

    # Drop rows with NaN values
    results_df.dropna(inplace=True)

    # Use the original Beta and R-Squared values for clustering
    features = results_df[['Beta', 'R-Squared']].values

    # Determine the optimal number of clusters
    optimal_clusters = determine_optimal_clusters(features)

    # Perform Gaussian Mixture Model clustering with the optimal number of clusters
    gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
    cluster_labels = gmm.fit_predict(features)
    results_df['Cluster'] = cluster_labels

    # Save the results to a CSV file
    results_df.to_csv('Israel_stocks_results.csv', index=False)

    # Generate random colors for each cluster
    custom_colors = []
    for _ in range(optimal_clusters):
        r = random.random()
        g = random.random()
        b = random.random()
        custom_colors.append(f'rgb({r}, {g}, {b})')

    # Create traces for each cluster
    traces = []
    for cluster in sorted(results_df['Cluster'].unique()):
        cluster_df = results_df[results_df['Cluster'] == cluster]
        trace = go.Scatter(x=cluster_df['Beta'], y=cluster_df['R-Squared'],
                           mode='markers', marker=dict(size=cluster_df['Latest Close'],
                                                       sizeref=2. * max(cluster_df['Latest Close']) / (60. ** 2),
                                                       sizemode='area'),
                           hovertext=cluster_df['Symbol'] + '<br>' + cluster_df['Name'] + '<br>Beta: ' + cluster_df[
                               'Beta'].astype(str) + '<br>R-Squared: ' + cluster_df['R-Squared'].astype(
                               str) + '<br>Latest Close: ' + cluster_df['Latest Close'].astype(str),
                           name=f'Cluster {cluster}',
                           marker_color=custom_colors[cluster % len(custom_colors)])
        traces.append(trace)

    # Create the layout
    layout = go.Layout(title=f'TASE Tel Aviv 125 Index Stock Clustering based on Beta and R-Squared (Optimal Clusters: {optimal_clusters})',
                       xaxis=dict(title='Beta', tickmode='linear', dtick=0.25),
                       yaxis=dict(title='R-Squared'))

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)

    # Add red lines at beta = 0 and beta = 1
    fig.add_trace(go.Scatter(x=[0, 0], y=[results_df['R-Squared'].min(), results_df['R-Squared'].max()],
                             mode="lines", line=dict(color="black", width=2), showlegend=False))
    fig.add_trace(go.Scatter(x=[1, 1], y=[results_df['R-Squared'].min(), results_df['R-Squared'].max()],
                             mode="lines", line=dict(color="black", width=2), showlegend=False))

    # Display the number of valid stocks and the latest date of data
    latest_date = end_date.strftime('%d/%m/%Y')
    fig.add_annotation(x=results_df['Beta'].max(), y=results_df['R-Squared'].max(),
                       text=f"Valid Stocks: {valid_stocks_count} | Date: {latest_date}",
                       showarrow=False, font=dict(size=14, color="black"))

    fig.show()

    print(f"Number of valid stocks: {valid_stocks_count}")

stocks_filepath = "TASE_stock_list_2023.csv"
index_symbol = "^TA125.TA"  # Tel Aviv 125 Index
analyze_israeli_stocks(stocks_filepath, index_symbol, years=5)
