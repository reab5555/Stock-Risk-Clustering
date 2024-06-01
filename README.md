# Stock-Risk-Clustering
This project aims to cluster stocks from the S&P 500 index based on their beta values and R-squared values in order to asses risks for investment portfolios. The script fetches historical stock data from Yahoo Finance, calculates the beta and R-squared for each stock against the S&P 500 index, and performs clustering using the Gaussian Mixture Model (GMM) algorithm.    

## Beta and R-Squared in Finance

### Beta
Beta is a measure of a stock's volatility in relation to the overall market. It indicates how sensitive a stock's price is to market movements.

* β = 1: Indicates that the stock's price will move with the market.   
* β < 1: Indicates that the stock is less volatile than the market.   
* β > 1: Indicates that the stock is more volatile than the market.   
* β = 0: Indicates that the stock's price is not correlated with the market.   
* Negative β: Indicates that the stock moves inversely to the market.   

### R-Squared
R-squared, also known as the coefficient of determination, is a statistical measure that represents the proportion of the variance in a dependent variable that can be explained by an independent variable. In the context of finance, R-squared is often used to evaluate how well a stock's price movements can be explained by the movements of the overall market. A higher R-squared value indicates a stronger relationship between the stock and the market.

## Project Overview

The script performs the following steps:

1. Fetches historical stock data for the S&P 500 constituents from Yahoo Finance.
2. Calculates the beta and R-squared values for each stock against the S&P 500 index.
3. Standardizes the beta and R-squared values.
4. Determines the optimal number of clusters using the Bayesian Information Criterion (BIC).
5. Performs Gaussian Mixture Model clustering with the optimal number of clusters.
6. Filters out outliers for plotting purposes.
7. Generates a scatter plot using Plotly, where each point represents a stock, colored according to its assigned cluster.
   - The x-axis represents the beta values.
   - The y-axis represents the R-squared values.
   - The size of each point represents the latest close price of the stock.
8. Adds red lines at beta = 0 and beta = 1 for reference.
9. Displays the number of valid stocks used in the analysis.

## Dependencies

The script requires the following Python libraries:

- pandas
- numpy
- tqdm
- scikit-learn
- plotly
- matplotlib
- yfinance

You can install these dependencies using pip:

```
pip install pandas numpy tqdm scikit-learn plotly matplotlib yfinance
```

## Usage

1. Ensure that you have the required dependencies installed.
2. Prepare a CSV file containing the list of S&P 500 stock symbols. The file should have a column named "Symbol".
3. Update the `stocks_filepath` variable in the script with the path to your CSV file.
4. Run the script:

```
python stock_clustering.py
```

5. The script will fetch the stock data, perform the clustering analysis, and display the resulting plot.

## Results

The script generates a scatter plot showing the clustered stocks based on their beta and R-squared values. Each cluster is represented by a different color, and the size of each point indicates the latest close price of the stock. The plot also includes red lines at beta = 0 and beta = 1 for reference.

The number of valid stocks used in the analysis is displayed in the plot and printed in the console output.

Feel free to explore and modify the script to suit your specific requirements.
