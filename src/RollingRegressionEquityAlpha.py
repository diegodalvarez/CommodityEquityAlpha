# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:47:44 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

class RollingRegressionAlpha:
    
    def __init__(self) -> None: 
        
        self.path       = os.getcwd()
        self.root_path  = os.path.abspath(os.path.join(self.path, os.pardir))
        self.data_path  = os.path.join(self.root_path, "data")
        self.ols_window = 30
        self.window     = 5
        
    def _get_eq_prep(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "eq_prep.parquet")
        if os.path.exists(out_path) == True: 
            if verbose: print("Equity Prep already generated")
            return None
        
        eq_path    = os.path.join(self.data_path, "eq_px.parquet")
        df_eq_prep = (pd.read_parquet(
            path = eq_path, engine = "pyarrow").
            query("variable == 'Adj Close'").
            pivot(index = "date", columns = "ticker", values = "value").
            pct_change().
            reset_index().
            melt(id_vars = ["date", "SPY"]).
            dropna())
        
        if verbose: print("Saving Eq prep")
        df_eq_prep.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_single_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_out = (RollingOLS(
            endog  = df.value,
            exog   = sm.add_constant(df.SPY),
            window = self.ols_window).
            fit().
            params.
            drop(columns = ["SPY"]).
            rename(columns = {"const": "eq_alpha"}).
            dropna())
        
        return df_out
    
    def _get_multi_alpha(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (RollingOLS(
            endog  = df.value,
            exog   = sm.add_constant(df[["SPY", "DXY", "TNX"]]),
            window = self.ols_window).
            fit().
            params.
            drop(columns = ["SPY", "TNX", "DXY"]).
            rename(columns = {"const": "eq_alpha"}).
            dropna())
        
        return df_out
        
    def _get_rolling_regression(self, verbose: bool = True) -> None: 
        
        
        out_path = os.path.join(self.data_path, "RollingEqAlpha.parquet")
        if os.path.exists(out_path) == True: 
            if verbose: print("Rolling Equity Alphas already generated")
            return None
        
        ticker_path = os.path.join(self.data_path, "tickers.xlsx")
        tickers     = (pd.read_excel(
            io = ticker_path, sheet_name = "Sheet2").
            Ticker.
            drop_duplicates().
            to_list())
        
        if verbose: print("Calculating Rolling Equity Alphas")
        
        eq_path = os.path.join(self.data_path, "eq_prep.parquet")
        df_eq   = (pd.read_parquet(
            path = eq_path, engine = "pyarrow").
            set_index("date").
            query("ticker == @tickers"))
        
        macro_path = os.path.join(self.data_path, "RawMacroFactors.parquet")
        df_macro   = (pd.read_parquet(
            path = macro_path, engine = "pyarrow").
            assign(
                DXY = lambda x: x.DXY.pct_change(),
                TNX = lambda x: np.log(x.TNX).diff()).
            dropna())
        
        df_wider = (df_eq.reset_index().pivot(
            index = ["date", "SPY"], columns = "ticker", values = "value").
            reset_index().
            set_index("date"))
        
        df_risk_adj = (df_wider.apply(
            lambda x: x / x.ewm(span = self.window, adjust = False).std()).
            reset_index().
            melt(id_vars = ["date", "SPY"]).
            set_index("date"))
        
        df_zscore = (df_wider.reset_index().set_index(
            ["date", "SPY"]).
            apply(lambda x: (
                x - x.ewm(span = self.window, adjust = False).mean()) / 
                x.ewm(span = self.window, adjust = False).std()).
            reset_index().
            assign(SPY = lambda x: x.SPY / x.SPY.ewm(span = self.window, adjust = False).std()).
            melt(id_vars =  ["date", "SPY"]).
            dropna())
        
        df_model1 = (df_eq.groupby(
            "ticker").
            apply(self._get_single_alpha).
            reset_index().
            assign(group = "model1"))
        
        df_model2 = (df_eq.merge(
            right = df_macro, how = "inner", on = ["date"]).
            groupby("ticker").
            apply(self._get_multi_alpha).
            reset_index().
            assign(group = "model2"))
        
        df_model3 = (df_risk_adj.groupby(
            "ticker").
            apply(self._get_single_alpha).
            reset_index().
            assign(group = "model3"))
        
        df_model4 = (df_eq.reset_index().pivot(
            index = ["date", "SPY"], columns = "ticker", values = "value").
            reset_index().
            set_index("date").
            apply(lambda x: x / x.ewm(span = self.window, adjust = False).std()).
            reset_index().
            melt(id_vars = ["date", "SPY"]).
            dropna().
            merge(right = df_macro, how = "inner", on = ["date"]).
            set_index("date").
            groupby("ticker").
            apply(self._get_multi_alpha).
            reset_index().
            assign(group = "model4"))
        
        df_model5 = (df_zscore.set_index(
            "date").
            groupby("ticker").
            apply(self._get_single_alpha).
            reset_index().
            assign(group = "model5"))

        df_model6 = (df_zscore.merge(
            right = df_macro, how = "inner", on = ["date"]).
            set_index("date").
            groupby("ticker").
            apply(self._get_multi_alpha).
            reset_index().
            assign(group = "model6"))
        
        df_out = (pd.concat([
            df_model1, df_model2, df_model3, 
            df_model4, df_model5, df_model6]))
        
        if verbose: print("Saving Rolling Equity Alpha\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
        
    rolling_regression = RollingRegressionAlpha()
    rolling_regression._get_eq_prep()
    rolling_regression._get_rolling_regression()
    
if __name__ == "__main__":  main()