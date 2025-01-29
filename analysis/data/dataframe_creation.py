import pandas as pd
import pandas_datareader as dr
import numpy as np 
import json
import yfinance as yf
from analysis.data.utils_analysis import create_full_dataset
import quantstats as qs
import warnings
warnings.filterwarnings("ignore")

def data_vertical(data):
    dataset_vertical = data.stack(level=1).reset_index()
    dataset_vertical.rename(columns={'level_1': 'Ticker'}, inplace=True)
    dataset_vertical['Date'] = pd.to_datetime(dataset_vertical['Date']) 
    dataset_vertical.set_index('Date', inplace=True)
    return dataset_vertical
    


def feature_engineering(data, rf, mkt):
    #mkt correlation
    rets = data["Close"].pct_change().dropna()
    correlations = np.array(rets.corrwith(mkt["SPY"].pct_change().dropna()))



    #CAPM beta
    merged_returns = rets.join(mkt["SPY"].pct_change().dropna(), how="inner") 

    # Compute covariance of each asset with SPY
    cov_with_spy = merged_returns.cov().iloc[:-1, -1] 

    # Compute variance of SPY
    spy_variance = merged_returns["SPY"].var()

    # Compute beta for each asset
    beta = cov_with_spy / spy_variance
    beta.name = "Beta"
    beta = np.array(beta)
    returns = data["Close"].pct_change().mean()*252
    returns = pd.DataFrame(returns)
    final_dataframe = returns.reset_index()
    final_dataframe = final_dataframe.rename(columns={final_dataframe.columns[1]:"Yavg_return"})
    #volatility
    final_dataframe["Yavg_volatility"] = np.array(data["Close"].pct_change().std()*np.sqrt(252))
    #CAPM beta
    final_dataframe["beta"] = beta
    #mkt corr
    final_dataframe["mkt_corr"] = correlations

    
    dataset_vertical = data_vertical(data)
    dataset_vertical["daily_span"] = dataset_vertical["High"]- dataset_vertical["Low"]
    #daily_span
    final_dataframe["Davg_span"] = np.array(dataset_vertical.groupby("Ticker")["daily_span"].mean())
    #traded volume
    final_dataframe["Davg_volume"] = np.array(dataset_vertical.groupby("Ticker")["Volume"].mean())
    #skewness
    final_dataframe["D_eSkewness"] = np.array(dataset_vertical.groupby("Ticker")["Close"].apply(qs.stats.skew))

    #VaR
    final_dataframe["D_eVaR"] = np.array(dataset_vertical.groupby("Ticker")["Close"].apply(qs.stats.value_at_risk))/(np.array(dataset_vertical.groupby("Ticker")["Close"].mean()))
    #CVaR
    final_dataframe["D_eCVaR"] = np.array(dataset_vertical.groupby("Ticker")["Close"].apply(qs.stats.expected_shortfall))/(np.array(dataset_vertical.groupby("Ticker")["Close"].mean()))
    #Curtosis
    final_dataframe["D_eCurtosis"] = np.array(dataset_vertical.groupby("Ticker")["Close"].apply(qs.stats.kurtosis))
    final_dataframe["Sharpe_ratio"] = (np.array(final_dataframe["Yavg_return"]) - rf)/np.array(final_dataframe["Yavg_volatility"])

    
    return final_dataframe




def pipeline(start_date, end_date, rf = 0.02):
    snp500url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    data_tab = pd.read_html(snp500url)
    market = yf.download(tickers="SPY", start=start_date, end=end_date, auto_adjust=True)["Close"]

    tickers = data_tab[0][1:]['Symbol'].tolist()

    print("Total number of tickers", len(tickers))

    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    location = r"C:\Users\m.narese\Desktop\THESIS\REPO\portfolio_optimization\analysis\datasets\esg_data.json"

    with open(location, "r") as file:
        esg_data = json.load(file)  # Parse JSON into a Python dictionary or list

    rows = []
    for entry in esg_data:
        ticker = entry['ticker']
        esgScores = entry.get('esgScores')
        if esgScores:
            rows.append({
                'ticker': ticker,
                'totalEsg': esgScores.get('totalEsg'),
                'environmentScore': esgScores.get('environmentScore'),
                'socialScore': esgScores.get('socialScore'),
                'governanceScore': esgScores.get('governanceScore')
            })
        else:
            rows.append({
                'ticker': ticker,
                'totalEsg': None,
                'environmentScore': None,
                'socialScore': None,
                'governanceScore': None
            })
    # Create DataFrame
    esg_df = pd.DataFrame(rows)

    dataset = pd.DataFrame(raw)
    missing_frac = dataset.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_frac[missing_frac > 0.2].index))
    dataset.drop(columns=drop_list, axis = 1, inplace=True)
    dataset.bfill(axis='index', inplace=True)
    print('\nNull values:', dataset.isnull().values.any())
    print('NaN values:', dataset.isna().values.any())
    print("\nCreating features")
    fdata = feature_engineering(dataset, rf, market)

    ESG = pd.read_csv(r"C:\Users\m.narese\Desktop\THESIS\REPO\portfolio_optimization\analysis\datasets\1\ESG_data.csv")
    stock_data = create_full_dataset(fdata, esg_df)
    stock_data = create_full_dataset(stock_data, ESG)
    stock_data = stock_data.drop(columns=["logo", "name", "weburl", "exchange", "last_processing_date", "cik", "currency", 
                                        "environment_grade",
                                        "environment_level",
                                        "social_grade",
                                        "social_level",
                                        "governance_grade",
                                        "governance_level",
                                        "environment_score",
                                        "social_score",
                                        "governance_score",
                                        "total_score",
                                        "total_grade",
                                        "total_level"])
    print(f"The dataset has {stock_data.shape[0]} assets")
    print(f"The dataset has {stock_data.shape[1]-1} predictors:")
    for i in stock_data.columns:
        print(i)
    print("\n\nDataset creation finished\n")
    return stock_data


def create_portfolio_clustered(start_date, end_date, segments_df, tickers, w = "uniform"):

    portfolio_data_trained = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]


    prices_dataset = pd.DataFrame(portfolio_data_trained)
    missing_frac = prices_dataset.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_frac[missing_frac > 0.1].index))
    prices_dataset.drop(columns=drop_list, axis = 1, inplace=True)
    prices_dataset.bfill(axis='index', inplace=True)
    print('Null values:', prices_dataset.isnull().values.any())
    prices_dataset = prices_dataset.reset_index()

    for asset in prices_dataset.columns[1:]:
        prices_dataset[asset] = prices_dataset[asset].pct_change()
    prices_dataset = prices_dataset.dropna()

    portfolio_dataset = pd.DataFrame(portfolio_data_trained)
    missing_frac = portfolio_dataset.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_frac[missing_frac > 0.1].index))
    portfolio_dataset.drop(columns=drop_list, axis = 1, inplace=True)
    portfolio_dataset.bfill(axis='index', inplace=True)
    print('Null values:', portfolio_dataset.isnull().values.any())
    ticker_segments = pd.DataFrame({
    "Ticker": segments_df["Ticker"],
    "Sector": segments_df["Sector"],
    "Sharpe_ratio": segments_df["Sharpe_ratio"]
    })

    # 1. Reshape daily_prices to long format
    portfolio_dataset_long = portfolio_dataset.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Close")

    # 2. Merge with ticker_segments to get segment information
    merged_data = pd.merge(portfolio_dataset_long, ticker_segments, on="Ticker")

    # 3. Calculate daily returns for each ticker
    merged_data["Daily_Return"] = merged_data.groupby("Ticker")["Close"].pct_change()
    merged_data = merged_data.dropna()
    # 4. Group by Date and K_means_segments to calculate equally weighted portfolio returns
    if w == "sharpe":
        portfolio_returns = (
            merged_data.groupby(["Date", "Sector"]).apply(
                lambda x: np.average(x["Daily_Return"], weights=x["Sharpe_ratio"])
            ).reset_index()
        )

        portfolio_returns=portfolio_returns.rename(columns={0:"Daily_Return"})
    else:
        portfolio_returns = (
            merged_data.groupby(["Date", "Sector"])["Daily_Return"]
            .mean()  # Equally weighted
            .reset_index()
        )

    # 5. Pivot the data for better visualization (optional)
    portfolio_returns = portfolio_returns.pivot(
        index="Date", columns="Sector", values="Daily_Return"
    ).dropna()

    return portfolio_returns, prices_dataset