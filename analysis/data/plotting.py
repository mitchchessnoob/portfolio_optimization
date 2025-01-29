import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

def plot_kmeans(n_runs,
                cluster_range,
                meanDistortions,
                sil_score,
                inertias    ):
    plt.figure(figsize=(10,20))
    plt.subplot(3,1,1)
    for i in range(n_runs): 
        plt.plot(cluster_range, meanDistortions[i], color=f"C{i}", alpha = 0.1)
    plt.xlabel("k")
    plt.ylabel("Average Distortion")
    plt.xticks(cluster_range)
    plt.title("Selecting k with the Elbow Method", fontsize=20)
    plt.plot(cluster_range, np.mean(np.array(meanDistortions), axis=0), label="Mean distortion score")
    plt.legend()
    plt.subplot(3,1,2)
    for i in range(n_runs): 
        plt.plot(cluster_range, sil_score[i], color=f"C{i}", alpha = 0.1)
    plt.ylabel("Silhouette score")
    # plt.ylim(0.1, 0.3)
    plt.xticks(cluster_range)
    plt.plot(cluster_range, np.mean(np.array(sil_score), axis=0), label="Mean silhouette score")
    plt.legend()
    plt.subplot(3,1,3)
    for i in range(n_runs): 
        plt.plot(cluster_range, inertias[i], color=f"C{i}", alpha = 0.1)
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.xticks(cluster_range)
    plt.title("Selecting k with the Elbow Method", fontsize=20)
    plt.plot(cluster_range, np.mean(np.array(inertias), axis=0), label="Mean inertia score")
    plt.legend()
    plt.show()

def plot_cluster_industry(max_labels, dataframe, cluster_summary):
    colormap = plt.get_cmap('Pastel1')
    # Get unique sectors
    sectors = sorted(dataframe["Sector"].unique())
    i = 1

    plt.figure(figsize=(20, 20))

    for sector in sectors:
        # Count industries in the current sector
        focus = dataframe[dataframe["Sector"] == sector]["industry"].value_counts()
        
        # Group "Others" if there are too many industries
        if len(focus) > max_labels:
            others = pd.Series(focus[max_labels:].sum(), index=["Others"])
            focus = pd.concat([focus[:max_labels], others])
        num_labels = len(focus)
        colors = colormap(np.linspace(0, 1, num_labels))
        # Plot the pie chart
        plt.subplot(3, 3, i)
        wedges, texts, autotexts =  plt.pie(
            focus.values,
            labels=focus.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            normalize=True

        )
        

    # Change color of percentage labels to red
        # for autotext in autotexts:
        #     autotext.set_color('white')
        sr = cluster_summary[cluster_summary.reset_index()['Sector'] == sector].reset_index()['Sharpe_ratio'].values[0]
        plt.title(f"Sector {sector} with sharpe ratio {np.round(sr, 3)}")
        i += 1

    plt.tight_layout()
    plt.show()

def portfolio_evolution_time(portfolio_returns, benchmark):
    portfolio_evolution = pd.DataFrame(portfolio_returns["Date"])
    plt.figure(figsize=(12, 6))
    portfolio_evolution["SPY"] = (1 + benchmark["SPY"].pct_change()).cumprod()
    for portfolio in portfolio_returns.columns:
        if portfolio != "Date":
            portfolio_evolution[portfolio] = (1 + portfolio_returns.reset_index()[portfolio]).cumprod()
            plt.plot(portfolio_evolution["Date"], portfolio_evolution[portfolio], label = portfolio, color = f"C{int(portfolio)}", alpha = 1, linewidth=0.8)
    plt.plot(portfolio_evolution["Date"], portfolio_evolution["SPY"], label = "Benchmark", color="black")#, color = f"C{int(portfolio)+1}", linewidth=2)
    plt.hlines(1, xmin=min(portfolio_evolution["Date"]), xmax=max(portfolio_evolution["Date"]), color = "grey")
    plt.legend()
    plt.show()

def portfolio_span(asset_returns, portfolio_returns, dataframe, benchmark, sectors):
    portfolio_evolution = pd.DataFrame(asset_returns["Date"])

    plt.figure(figsize=(15, 30))

    # Dictionary to store portfolio cumulative returns
    portfolio_groups = {}
    portfolio_evolution["SPY"] = (1 + benchmark.reset_index()["SPY"].pct_change()).cumprod()
    for portfolio in asset_returns.columns:
        if portfolio != "Date":
            # Get portfolio label from clustering dataframe
            lab = dataframe.loc[dataframe["Ticker"] == portfolio, "Sector"].values[0]
            
            # Compute cumulative returns
            portfolio_evolution[portfolio] = (1 + asset_returns[portfolio]).cumprod()
            
            # Store returns grouped by portfolio
            if lab not in portfolio_groups:
                portfolio_groups[lab] = []
            portfolio_groups[lab].append(portfolio_evolution[portfolio])

    # Convert grouped lists to DataFrames
    for lab, assets in portfolio_groups.items():
        plt.subplot(sectors, 1, int(lab)+1)
        asset_group_df = pd.concat(assets, axis=1)
        
        # Compute min and max across assets in the portfolio
        min_returns = asset_group_df.min(axis=1)
        max_returns = asset_group_df.max(axis=1)
        
        # Shade area between min and max
        plt.fill_between(portfolio_evolution["Date"], min_returns, max_returns, color=f"C{int(lab)}", alpha=0.2)

    # Plot each portfolio's average return
    for portfolio in portfolio_returns.reset_index().columns:
        if portfolio != "Date":
            plt.subplot(sectors, 1, int(portfolio)+1)
            portfolio_evolution[portfolio] = (1 + portfolio_returns.reset_index()[portfolio]).cumprod()
            plt.plot(portfolio_evolution["Date"], portfolio_evolution[portfolio], label=portfolio, color=f"C{int(portfolio)}", alpha=1, linewidth=0.8)
            plt.plot(portfolio_evolution["Date"], portfolio_evolution["SPY"], label="Benchmark", color="black", linewidth=1.2)
            plt.legend()
            plt.hlines(1, xmin=min(portfolio_evolution["Date"]), xmax=max(portfolio_evolution["Date"]), color="grey", linestyle="--")


    # Plot the benchmark (SPY)



    # Add horizontal line at 1
    plt.hlines(1, xmin=min(portfolio_evolution["Date"]), xmax=max(portfolio_evolution["Date"]), color="grey", linestyle="--")

    plt.ylim(0, 7.5)
    plt.legend()
    plt.show()