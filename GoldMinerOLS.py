# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 13:01:58 2026

@author: Diego
"""

import os
import pandas as pd

from tqdm import tqdm

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

tqdm.pandas()

class GoldMiningOLS:
    
    def __init__(self, data_path: str) -> None: 
        
        self.data_path   = data_path
        self.ticker_path = os.path.join(self.data_path, "tickers.xlsx")
        
        self.df_tickers = (pd.read_excel(
            io         = self.ticker_path,
            sheet_name = "gdx_tickers")
            [["country", "yf_ticker", "benchmark", "country_benchmark", "country_type"]])
        
    def _get_alpha(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        
        df_tmp  = df.dropna().sort_index()
        df_out = (RollingOLS(
            endog  = df_tmp.single_rtn,
            exog   = sm.add_constant(df_tmp.exog_rtn),
            window = window).
            fit().
            params.
            rename(columns = {
                "exog_rtn": "beta",
                "const"   : "alpha"}).
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out
    
    def _get_multi_alpha(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        
        df_tmp       = df.dropna().sort_index()
        exog_tickers = [ticker for ticker in df_tmp.columns.to_list() if ticker != "endog_rtn"]
        
        df_out = (RollingOLS(
            endog  = df_tmp.endog_rtn,
            exog   = sm.add_constant(df_tmp[exog_tickers]),
            window = window).
            fit().
            params.
            rename(columns = {"const": "alpha"}).
            reset_index().
            melt(id_vars = ["date", "alpha"]).
            assign(variable = lambda x: x.variable + "_beta").
            pivot(index = ["date", "alpha"], columns = "variable", values = "value").
            reset_index().
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out
    
    def run_ols(self, window: int = 30, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "GoldMinersAlpha.parquet")
        if os.path.exists(out_path) == False: 
        
            if verbose: print("Generating Gold Mining Equity Alpha\n")    
        
            raw_path = os.path.join(self.data_path, "GoldMiners.parquet")    
            etf_path = os.path.join(self.data_path, "eq_px.parquet")
            
            df_raw = (pd.read_parquet(
                path = raw_path).
                reset_index())
            
            df_rtn = (df_raw.assign(
                date = lambda x: pd.to_datetime(x.Date).dt.tz_convert("America/New_York").dt.date).
                pivot(index = "date", columns = "ticker", values = "Adj Close").
                pct_change().
                reset_index().
                melt(id_vars = "date", value_name = "rtn").
                dropna())
            
            # for SPY test
            df_spy = (df_rtn.assign(
                front = lambda x: x.ticker.str[0]).
                query("front != '^'").
                pivot(index = "date", columns = "ticker", values = "rtn").
                reset_index().
                melt(id_vars = ["date", "SPY"], value_name = "single_rtn").
                dropna().
                assign(group = "spy").
                rename(columns = {"SPY": "exog_rtn"}).
                assign(exog_ticker = "SPY"))
            
            # for Developing vs. Non Developing Test
            df_tmp = (self.df_tickers[
                ["yf_ticker", "country_benchmark"]].
                rename(columns = {"yf_ticker": "ticker"}))
            
            df_country_rtn = (df_rtn.rename(columns = {
                "ticker": "country_benchmark",
                "rtn"   : 'country_rtn'}))
            
            df_country_type = (df_rtn.merge(
                right = df_tmp, how = "inner", on = ["ticker"]).
                rename(columns = {"rtn": "single_rtn"}).
                merge(right = df_country_rtn, how = "inner", on = ["country_benchmark", "date"]).
                rename(columns = {
                    "country_benchmark": "exog_ticker",
                    "country_rtn"      : "exog_rtn"}).
                assign(group = "country_type_group"))
            
            # for each country test
            df_country = (df_rtn.rename(
                columns = {"ticker": "yf_ticker"}).
                merge(right = self.df_tickers, how = "inner", on = ["yf_ticker"])
                [["date", "yf_ticker", "rtn", "benchmark"]].
                rename(columns = {"rtn": "single_rtn"}).
                rename(columns = {"benchmark": "ticker"}).
                merge(right = df_rtn, how = "inner", on = ["date", "ticker"]).
                rename(columns = {
                    "yf_ticker": "ticker",
                    "ticker"   : "exog_ticker",
                    "rtn"      : "exog_rtn"}).
                assign(group = "country"))
            
            df_gdx = (pd.read_parquet(
                path = etf_path, engine = "pyarrow").
                query("ticker == 'GDX' & variable == 'Adj Close'")
                [["date", "value"]])
            
            tickers    = self.df_tickers.yf_ticker.drop_duplicates().sort_values().to_list()
            df_gdx_tmp = (df_rtn.query(
                "ticker == @tickers").
                merge(right = df_gdx, how = "inner", on = ["date"]).
                rename(columns = {
                    "rtn"  : "single_rtn",
                    "value": "exog_rtn"}).
                assign(
                    exog_ticker = "gdx",
                    group       = "gdx"))
            
            df_combined = pd.concat([df_country, df_country_type, df_spy, df_gdx_tmp])
            
            df_ols = (df_combined.assign(
                group_var = lambda x: x.ticker + " " + x.group).
                set_index("date").
                groupby("group_var").
                progress_apply(lambda group: self._get_alpha(group, window)))
            
            if verbose: print("\nSaving data")
            df_ols.to_parquet(path = out_path, engine = "pyarrow")
            
        else: 
            if verbose: print("Already Saved Single Endog Data")
            
    def run_multi_ols(self, window: int = 30, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "MultiGoldMinersAlpha.parquet")
        if os.path.exists(out_path) == False: 
        
            if verbose: print("Generating Gold Mining Equity Alpha\n")    
        
            raw_path = os.path.join(self.data_path, "GoldMiners.parquet")    
            etf_path = os.path.join(self.data_path, "eq_px.parquet")
            
            df_raw = (pd.read_parquet(
                path = raw_path).
                reset_index())
            
            df_rtn = (df_raw.assign(
                date = lambda x: pd.to_datetime(x.Date).dt.tz_convert("America/New_York").dt.date).
                pivot(index = "date", columns = "ticker", values = "Adj Close").
                pct_change().
                reset_index().
                melt(id_vars = "date", value_name = "rtn").
                dropna())
            
            df_exog_tickers = (self.df_tickers[
                ["benchmark", "country_benchmark", "yf_ticker"]].
                assign(gdx = "GDX").
                melt(id_vars = "yf_ticker").
                drop(columns = ["yf_ticker"]).
                drop_duplicates())
            
            exog_tickers = df_exog_tickers.value.drop_duplicates().to_list()
            
            df_exog = (df_rtn.query(
                "ticker == @exog_tickers").
                assign(ticker = lambda x: x.ticker.str.replace("^", "")).
                pivot(index = "date", columns = "ticker", values = "rtn").
                dropna())
            
            df_endog = (df_rtn.query(
                "ticker != @exog_tickers").
                rename(columns = {
                    "ticker": "endog_ticker",
                    "rtn"   : "endog_rtn"}))
            
            df_ols = (df_endog.merge(
                right = df_exog, how = "inner", on = ["date"]).
                set_index("date").
                groupby("endog_ticker").
                apply(self._get_multi_alpha, include_groups = False).
                reset_index().
                drop(columns = ["level_1"]))
            
            if verbose: print("\nSaving data")
            df_ols.to_parquet(path = out_path, engine = "pyarrow")
            
        else: 
            if verbose: print("Already Saved Multi Exog Data")

data_path       = os.path.join(os.getcwd(), "data")
gold_mining_ols = GoldMiningOLS(data_path)
#gold_mining_ols.run_ols()
gold_mining_ols.run_multi_ols()