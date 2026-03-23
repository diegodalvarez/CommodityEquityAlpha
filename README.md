# Commodity Equity Alpha
This repo explores trading strategies regarding commodity futures based on the equity alpha of companies that are related to the underlying strategy. The motivation of these strategies is based on the premise that equity investors attribute positive/negative excess returns (alpha) to commodity companies who in return directly affect commodity prices. This relationship doesn't translate well into pure commodities and thus the inefficiency can be captured. 

The first set of strategies are developed using ETFs of commodity companies which can be found in ```1BaseModel.ipynb```

Below is the performance summary statistics for the model

| Commod Group            |   Perfect Vol. Target |   Lagged Vol. Target |
|:------------------------|----------------------:|---------------------:|
| Agribusiness            |                 1.244 |                1.272 |
| Copper and Green Metals |                 0.824 |                0.379 |
| Oil Refiners            |                 0.531 |                0.249 |
| Oil Services            |                 1.075 |                0.976 |
| Gold Miners             |                 0.962 |                0.791 |

The individual strategies are can be combined into a portfolio that generates a portfolio of around 1.15-1.2 sharpe. 

That model can be refined using single names stocks as well. The prior model looked at ETF alpha to trade commodities in ```2GoldModel.ipynb``` using the alpha of specific gold miners can achieve 0.8 - 1.15 sharpe. 
