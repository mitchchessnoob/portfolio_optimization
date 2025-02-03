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
    
    returns = data["Close"].pct_change().mean()*252
    returns = pd.DataFrame(returns)
    final_dataframe = returns.reset_index()
    final_dataframe = final_dataframe.rename(columns={final_dataframe.columns[1]:"Yavg_return"})
    #volatility
    final_dataframe["Yavg_volatility"] = np.array(data["Close"].pct_change().std()*np.sqrt(252))
    #mkt corr
    final_dataframe["mkt_corr"] = correlations

    #Last 1 year momentum
    latest_date = data.index.max()


    one_year_ago = latest_date - pd.DateOffset(years=1)


    df_last_year = data[data.index >= one_year_ago]["Close"]
    def compute_momentum(series):
        return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100

    final_dataframe["1Y_momentum"] = np.array(df_last_year.apply(compute_momentum))
    ####
    dataset_vertical = data_vertical(data)
    dataset_vertical["daily_span"] = dataset_vertical["High"]- dataset_vertical["Low"]
    #daily_span
    final_dataframe["Davg_span"] = np.array(dataset_vertical.groupby("Ticker")["daily_span"].mean())
    #traded volume
    final_dataframe["Davg_volume"] = np.array(dataset_vertical.groupby("Ticker")["Volume"].mean())
    #skewness and kurtosis
    dataset_vertical["Returns"] = dataset_vertical.groupby("Ticker")["Close"].pct_change()

    kurtosis_dict = dataset_vertical.groupby("Ticker")["Returns"].apply(lambda x: qs.stats.kurtosis(x.dropna()))  
    skewness_dict = dataset_vertical.groupby("Ticker")["Returns"].apply(lambda x: qs.stats.skew(x.dropna()))

    stats_df = pd.DataFrame({"Ticker": kurtosis_dict.index, "Davg_Kurtosis": kurtosis_dict.values, "Davg_Skewness": skewness_dict.values})

    final_dataframe = final_dataframe.merge(stats_df, on="Ticker")

    #VaR
    var_dict = dataset_vertical.groupby("Ticker")["Returns"].apply(lambda x: qs.stats.value_at_risk(x.dropna()))

    var_df = pd.DataFrame(var_dict).reset_index()
    var_df.columns = ["Ticker", "D_eVaR"]
    final_dataframe["D_eVaR"]  = var_df["D_eVaR"] 

    #CVaR
    cvar_dict = dataset_vertical.groupby("Ticker")["Returns"].apply(lambda x: qs.stats.expected_shortfall(x.dropna()))

    cvar_df = pd.DataFrame(cvar_dict).reset_index()
    cvar_df.columns = ["Ticker", "D_eCVaR"]

    final_dataframe["D_eCVaR"]  = cvar_df["D_eCVaR"]
    
    #Sharpe Ratio
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
        esg_data = json.load(file)  

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

    esg_df = pd.DataFrame(rows)

    location = r"C:\Users\m.narese\Desktop\THESIS\REPO\portfolio_optimization\analysis\datasets\company_data.json"

    with open(location, "r") as file:
        company_data = json.load(file)  

    rows = []
    for ticker in company_data:
        

        try:
            rows.append({
                'ticker': ticker,
                'beta': company_data[ticker].get('beta'),
                'ROA': company_data[ticker].get('returnOnAssets'),
                'ROE': company_data[ticker].get('returnOnEquity'),
                'est_ROI': company_data[ticker].get('netIncomeToCommon')/company_data[ticker].get('marketCap'),
                'profitMargins': company_data[ticker].get('profitMargins'),
                'P/B': company_data[ticker].get('priceToBook'),
                'earningsGrowth': company_data[ticker].get('earningsGrowth'),
                'forwardPE': company_data[ticker].get('forwardPE'),
            })
        except TypeError:
            print(f"Income. {company_data[ticker].get('netIncomeToCommon')}")
            print(f"MktCap. {company_data[ticker].get('marketCap')}")
            print(f"Ticker: {ticker}")
            rows.append({
                'ticker': ticker,
                'beta': company_data[ticker].get('beta'),
                'ROA': company_data[ticker].get('returnOnAssets'),
                'ROE': company_data[ticker].get('returnOnEquity'),
                'est_ROI': None,
                'profitMargins': company_data[ticker].get('profitMargins'),
                'P/B': company_data[ticker].get('priceToBook'),
                'earningsGrowth': company_data[ticker].get('earningsGrowth'),
                'forwardPE': company_data[ticker].get('forwardPE'),
            })
    
        
    company_df = pd.DataFrame(rows)

    dataset = pd.DataFrame(raw)
    missing_frac = dataset.isnull().mean().sort_values(ascending=False)
    drop_list = sorted(list(missing_frac[missing_frac > 0.2].index))
    print(f"\nThe following tickers had more than 20% of NaN values, therefore they're removed:")
    jj = set()
    for (_, i) in drop_list:
        jj.add(i)
    print(jj)
    dataset.drop(columns=drop_list, axis = 1, inplace=True)
    dataset.bfill(axis='index', inplace=True)
    print('\nNull values:', dataset.isnull().values.any())
    print('NaN values:', dataset.isna().values.any())
    print("\nCreating features")
    fdata = feature_engineering(dataset, rf, market)

    ESG = pd.read_csv(r"C:\Users\m.narese\Desktop\THESIS\REPO\portfolio_optimization\analysis\datasets\1\ESG_data.csv")
    stock_data = create_full_dataset(fdata, esg_df)
    stock_data = create_full_dataset(stock_data, ESG)
    stock_data = create_full_dataset(stock_data, company_df)
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
    print(f"The dataset has {stock_data.shape[1]-2} predictors:")
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
    "Sharpe_ratio": segments_df["Sharpe_ratio"],
    "volatility": segments_df["Yavg_volatility"]
    })

    portfolio_dataset_long = portfolio_dataset.reset_index().melt(id_vars="Date", var_name="Ticker", value_name="Close")

    merged_data = pd.merge(portfolio_dataset_long, ticker_segments, on="Ticker")

    merged_data["Daily_Return"] = merged_data.groupby("Ticker")["Close"].pct_change()
    merged_data = merged_data.dropna()

    if w == "sharpe":
        print("Cluster Portfolios based on sharpe ratio")
        portfolio_returns = (
            merged_data.groupby(["Date", "Sector"]).apply(
                lambda x: np.average(x["Daily_Return"], weights=x["Sharpe_ratio"])
            ).reset_index()
        )

        portfolio_returns=portfolio_returns.rename(columns={0:"Daily_Return"})

    elif w.lower() == "volatility":
        print("Cluster Portfolios based on volatility")
        portfolio_returns = (
            merged_data.groupby(["Date", "Sector"]).apply(
                lambda x: np.average(x["Daily_Return"], weights=1/x["volatility"])
            ).reset_index()
        )

        portfolio_returns=portfolio_returns.rename(columns={0:"Daily_Return"})
    else:
        print("Cluster Portfolios uniformly built among assets")
        portfolio_returns = (
            merged_data.groupby(["Date", "Sector"])["Daily_Return"]
            .mean()  # Equally weighted
            .reset_index()
        )

    portfolio_returns = portfolio_returns.pivot(
        index="Date", columns="Sector", values="Daily_Return"
    ).dropna()

    return portfolio_returns, prices_dataset