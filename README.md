# Stock-Risk-Clustering
This project aims to cluster stocks from the S&P 500 index or the Tel Aviv Stock Exchange (TASE) based on their beta values and R-squared values in order to asses risks for investment portfolios.  
   
The script fetches historical stock data from Yahoo Finance (a default period of five years), calculates the beta and R-squared for each stock against the S&P 500 index, and performs clustering using the Gaussian Mixture Model (GMM).    

## Beta and R-Squared in Finance

### Beta
Beta is a measure of a stock's volatility in relation to the overall market. It indicates how sensitive a stock's price is to market movements.

* β = 1: Indicates that the stock's price will move with the market.   
* β < 1: Indicates that the stock is less volatile than the market.   
* β > 1: Indicates that the stock is more volatile than the market.   
* β = 0: Indicates that the stock's price is not correlated with the market.   
* Negative β: Indicates that the stock moves inversely to the market.   

### R-Squared
In the context of finance, R-squared is often used to evaluate how well a stock's price movements can be explained by the movements of the overall market. A higher R-squared value indicates a stronger relationship between the stock and the market.

R² = 1: This indicates a perfect linear relationship between the stock and the market. All of the stock's price movements can be explained by the market's movements.   
R² = 0.5: This suggests that 50% of the stock's price movements can be explained by the market's movements, while the remaining 50% is due to other factors.   
R² = 0.25: This indicates that only 25% of the stock's price movements can be explained by the market's movements, while the remaining 75% is attributed to other factors.   
R² = 0: This suggests that none of the stock's price movements can be explained by the market's movements. The stock's price movements are entirely independent of the market.   

## Gaussian Mixture Model (GMM)
Gaussian Mixture Model (GMM) is a probabilistic model that assumes the data is generated from a mixture of a finite number of Gaussian distributions. It is commonly used for clustering tasks, where the goal is to group similar data points together based on their underlying probability distributions.    

The optimal number of clusters can be determined using the Bayesian Information Criterion (BIC), as it balance the goodness of fit with the complexity of the model, helping to avoid overfitting or underfitting.

## Dependencies

You can install these dependencies using pip:

```
pip install pandas numpy tqdm scikit-learn plotly matplotlib yfinance
```

## Usage

### S&P 500 Stocks

```
python Stocks_Risk_Clustering_SP500.py
```

### TASE Stocks

```
python Stocks_Risk_Clustering_TASE.py
```


## Results

The script generates a scatter plot showing the clustered stocks based on their beta and R-squared values. Each cluster is represented by a different color, and the size of each point indicates the latest close price of the stock. The plot also includes red lines at beta = 0 and beta = 1 for reference.

The number of valid stocks used in the analysis is displayed in the plot and printed in the console output.
