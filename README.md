# Commodity Equity Alpha
This repo explores trading strategies with commodity futures based on the equity alpha of companies who's cashflow is linked to the physical commodity. The motivation of these strategies is based on the premise that equity investors attribute positive/negative excess returns (alpha) to commodity companies who in return directly affect commodity prices. This relationship doesn't translate well into pure commodities and thus the inefficiency can be captured. 

The first set of strategies are developed using ETFs of commodity companies which can be found in ```1BaseModel.ipynb```

Below is the performance summary statistics for the model

| Name                    |   Perfect Vol. Target |   Lagged Vol. Target |
|:------------------------|----------------------:|---------------------:|
| Agribusiness            |                 1.25  |                1.279 |
| Copper and Green Metals |                 1.693 |                1.231 |
| Gold Miners             |                 0.969 |                0.792 |
| Natural Resources       |                 1.639 |                1.43  |
| Oil Refiners            |                 0.527 |                0.234 |
| Oil Services            |                 1.064 |                0.966 |

The individual strategies are can be combined into a portfolio that generates a portfolio of around 1.16-1.31 sharpe. 

That model can be refined using single names stocks as well. The prior model looked at ETF alpha to trade commodities in ```2GoldModel.ipynb``` using the alpha of specific gold miners can achieve 0.8 - 1.15 sharpe. 

## Writeup
|         | PDF          |
|---------|---------------------|
| Technical writeup containing methodology & results | <a href="CommodityEquityAlpha.pdf">![PDF](https://img.icons8.com/ios-filled/50/000000/pdf.png)</a> |