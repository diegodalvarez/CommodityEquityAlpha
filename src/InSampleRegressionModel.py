# -*- coding: utf-8 -*-
"""
Created on Fri May  1 13:34:19 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tqdm import tqdm
tqdm.pandas()

class InSampleRegressionDecile:
    
    def __init__(self) -> None:
        
        self.path       = os.getcwd()
        self.root_path  = os.path.abspath(os.path.join(self.path, os.pardir))
        self.data_path  = os.path.join(self.root_path, "data")
        
        self.lookback   = 100
        self.target     = 0.1
        self.model_dict = {
            "model1": "raw",
            "model4": "macro",
            "model6": "zscore"}
        
        self.q             = 10
        self.zscore_window = 30
        
    def _get_vol_targeted_commod(self) -> pd.DataFrame:
        
        fut_path = os.path.join(self.data_path, "commod_px.parquet")
        df_fut   = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            assign(security = lambda x: x.security.str.split("1").str[0].str.strip()).
            pivot(index = "date", columns = "security", values = "PX_LAST").
            pct_change().
            reset_index().
            melt(id_vars = ["date"], var_name = "fut_ticker", value_name = "fut_rtn").
            dropna().
            assign(date = lambda x: pd.to_datetime(x.date).dt.date))
        
        df_lagged_fut = (df_fut.pivot(
            index = "date", columns = "fut_ticker", values = "fut_rtn").
            apply(lambda x: x * (self.target / (x.ewm(span = self.lookback, adjust = False).std().shift() * np.sqrt(252)))).
            reset_index().
            melt(id_vars = "date", value_name = "lag_vol").
            dropna())
        
        df_perf_fut = (df_fut.pivot(
            index = "date", columns = "fut_ticker", values = "fut_rtn").
            apply(lambda x: x * (self.target / (x.ewm(span = self.lookback, adjust = False).std() * np.sqrt(252)))).
            reset_index().
            melt(id_vars = "date", value_name = "perf_vol").
            dropna())
        
        df_fut_prep = (df_fut.merge(
            right = df_lagged_fut, how = "inner", on = ["date", "fut_ticker"]).
            merge(right = df_perf_fut, how = "inner", on = ["date", "fut_ticker"]))
        
        return df_fut_prep
    
    def _prep_data(self, verbose: bool = True) -> None:
        
        out_path = os.path.join(self.data_path, "InSampleOLSSetup.parquet")
        if os.path.exists(out_path):
            if verbose: 
                print("Already Calculated In Sample OLS Setup")
            return None
        
        if verbose: 
            print("Preparing In-Sample OLS data")
        
        ticker_path = os.path.join(self.data_path, "Tickers.xlsx")
        df_ticker   = (pd.read_excel(
            io = ticker_path, sheet_name = "Sheet2")
            [["Ticker", "fut_ticker"]].
            rename(columns = {"Ticker": "etf_ticker"}))
        
        df_fut = (self._get_vol_targeted_commod().rename(
            columns = {"fut_rtn": "raw_rtn"}).
            melt(id_vars = ["date", "fut_ticker"], var_name = "rtn_calc", value_name = "fut_rtn"))
        
        alpha_path = os.path.join(self.data_path, "RollingEqAlpha.parquet")
        models     = list(self.model_dict.keys())
        df_alpha   = (pd.read_parquet(
            path = alpha_path, engine = "pyarrow").
            query("group == @models").
            replace(self.model_dict).
            set_index("date").
            groupby(["ticker", "group"]).
            apply(lambda x: x.shift()).
            reset_index().
            rename(columns = {
                "ticker"  : "etf_ticker",
                "eq_alpha": "lag_eq_alpha"}).
            dropna())
        
        df_combined = (df_alpha.merge(
            right = df_ticker, how = "inner", on = ["etf_ticker"]).
            merge(right = df_fut, how = "inner", on = ["date", "fut_ticker"]))
        
        if verbose: print("Saving data\n")
        df_combined.to_parquet(path = out_path, engine = "pyarrow")
        
    def _ols_get_resid(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df_signal = (sm.OLS(
            endog = df.fut_rtn,
            exog  = sm.add_constant(df.lag_eq_alpha)).
            fit().
            resid.
            to_frame(name = "resid").
            assign(zscore = lambda x: 
                   (x.resid - x.resid.ewm(span = self.zscore_window, adjust = False).mean()) /
                   x.resid.ewm(span = self.zscore_window, adjust = False).std()))
            
        df_decile = (df_signal.apply(
            lambda x: pd.qcut(x = x, q = self.q, labels = range(1, self.q + 1))).
            shift().
            rename(columns = {
                "resid" : "resid_decile",
                "zscore": "zscore_decile"}))
        
        df_out = (df_signal.shift().rename(
            columns = {
                "resid" : "lag_resid",
                "zscore": "lag_zscore"}).
            merge(right = df_decile, how = "inner", on = ["date"]).
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out

    def _run_is_resgression(self, verbose: bool = True) -> None: 
        
        in_path  = os.path.join(self.data_path, "InSampleOLSSetup.parquet")
        out_path = os.path.join(self.data_path, "ISRegression.parquet")
        df_setup = pd.read_parquet(path = in_path, engine = "pyarrow")
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have in-sample Regression Setup")
                
            return None
        
        if verbose:
            print("Working on in-sample regression setup")
        
        df_group_var = (df_setup[
            ["etf_ticker", "group", "fut_ticker", "rtn_calc"]].
            drop_duplicates().
            assign(group_var = lambda x: x.sum(axis = 1)))
        
        df_input = (df_setup.merge(
            right = df_group_var, 
            how   = "inner", 
            on    = ["etf_ticker", "group", "fut_ticker", "rtn_calc"]))
        
        df_out = (df_input.set_index(
            "date").
            groupby("group_var").
            progress_apply(lambda group: self._ols_get_resid(group)).
            reset_index())
        
        if verbose: 
            print("Saving data\n")
        
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _optimize_residuals(self, verbose: bool = True) -> None: 
        
        out_path = os.path.join(self.data_path, "ISResidOpt.parquet")
        in_path  = os.path.join(self.data_path, "ISRegression.parquet")
        
        if os.path.exists(out_path) == True: 
            if verbose:
                print("Already have in-sample optimized signal")
                
            return None
        
        if verbose: 
            print("Optimizing deciled in-sample regreession")
    
        df_decile = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            drop(columns = ["group_var"]).
            query("rtn_calc == 'raw_rtn' & group == 'zscore'").
            drop(columns = ["rtn_calc", "group", "lag_resid", "lag_zscore", "lag_eq_alpha"]).
            dropna().
            melt(
                id_vars    = ["date", "etf_ticker", "fut_ticker", "fut_rtn"],
                var_name   = "decile_group",
                value_name = "decile"))
        
        df_decile_sharpe = (df_decile.drop(
            columns = ["date"]).
            groupby(["etf_ticker", "fut_ticker", "decile_group", "decile"]).
            agg(lambda x: x.mean() / x.std() * np.sqrt(252)).
            reset_index().
            rename(columns = {"fut_rtn": "sharpe"}))
        
        df_decile_tmp = (df_decile_sharpe.query(
            "decile == [1,2,9,10]").
            assign(tmp_group = lambda x: np.where(x.decile <= 2, "lgroup", "ugroup")))
        
        df_out = (df_decile_tmp.drop(
            columns = ["decile"]).
            groupby(["etf_ticker", "fut_ticker", "tmp_group", "decile_group"]).
            agg("prod").
            reset_index().
            assign(signal_scaler = lambda x: np.where(x.sharpe > 0, 1, np.nan)).
            drop(columns = ["sharpe"]).
            merge(
                right = df_decile_tmp, 
                how   = "outer", 
                on    = ["etf_ticker", "fut_ticker", "tmp_group", "decile_group"]).
            merge(
                right = df_decile, 
                how   = "outer", 
                on    = ["etf_ticker", "fut_ticker", "decile", "decile_group"]).
            assign(signal_rtn = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fut_rtn))
        
        if verbose:
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 

    is_regress = InSampleRegressionDecile()        
    is_regress._prep_data()
    is_regress._run_is_resgression()
    is_regress._optimize_residuals()

if __name__ == "__main__": main()