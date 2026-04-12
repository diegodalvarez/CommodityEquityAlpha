# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:09:51 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from   statsmodels.regression.rolling import RollingOLS

from tqdm import tqdm

class OOSDecile:
    
    def __init__(self) -> None: 
        
        self.cur_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.cur_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        
    def _get_ols(self, df: pd.DataFrame, window: int) -> pd.DataFrame: 
        
        df_out = (RollingOLS(
            endog  = df.value,
            exog   = sm.add_constant(df.SPY),
            window = window).
            fit().
            params.
            rename(columns = {"SPY": "beta"}).
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out
        
    def _get_etf_alpha(self, verbose: bool = True, window: int = 30) -> None:
        
        '''
        Function that coculates n-days rolling alpha to SPY of the ETFs
        '''
        
        out_path = os.path.join(self.data_path, "ETFSPYAlpha.parquet")
        if os.path.exists(out_path):
            if verbose: 
                print("Already have ETF SPY Alpha")
            return None
            
            
        if verbose: 
            print("Getting ETF alpha data")
    
        eq_path  = os.path.join(self.data_path, "eq_px.parquet")
        df_alpha = (pd.read_parquet(
            path = eq_path, engine = "pyarrow").
            query("variable == 'Adj Close'").
            pivot(index = "date", columns = "ticker", values = "value").
            pct_change().
            reset_index().
            melt(id_vars = ["date", "SPY"]).
            dropna().
            set_index("date").
            groupby("ticker").
            apply(self._get_ols, window, include_groups = False).
            reset_index().
            dropna())
        
        if verbose:
            print("Saving data\n")
            
        df_alpha.to_parquet(path = out_path, engine = "pyarrow")
            
    def _oos_resid_calc(self, df: pd.DataFrame, min_nobs: int = 30) -> pd.DataFrame: 
        
        '''
        For computing the Expanding Out-of-Sample Residuals
        '''
        
    
        df_out = (RollingOLS(
            endog     = df.fut_rtn,
            exog      = sm.add_constant(df.lag_eq_alpha),
            expanding = True,
            min_nobs  = min_nobs).
            fit().
            params.
            rename(columns = {"lag_eq_alpha": "lag_alpha_beta"}).
            merge(right = df, how = "inner", on = ["date"]).
            assign(
                y_pred    = lambda x: (x.lag_alpha_beta * x.lag_eq_alpha) + x.const,
                resid     = lambda x: x.fut_rtn - x.y_pred,
                lag_resid = lambda x: x.resid.shift()))
    
        return df_out
        
    def _get_oos_resid(self, verbose: bool = True) -> None: 
        
        '''
        Function that calculates the OOS Residuals using the inputted alphas
        '''
        out_path = os.path.join(self.data_path, "OOSETFAlphaResid.parquet")
        
        if os.path.exists(out_path):
            if verbose:
                print("Already have Out-of-Sample Residuals")
            return None
        
        if verbose: print("Getting Out-of-Sample Residuals")
        
        fut_path   = os.path.join(self.data_path, "commod_px.parquet")
        df_fut_rtn = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            assign(security = lambda x: x.security.str.split("1").str[0].str.strip()).
            pivot(index = "date", columns = "security", values = "PX_LAST").
            pct_change().
            reset_index().
            melt(id_vars = ["date"], var_name = "fut_ticker", value_name = "fut_rtn").
            dropna().
            assign(date = lambda x: pd.to_datetime(x.date).dt.date))
        
        relationship_path = os.path.join(self.data_path, "tickers.xlsx")
        df_relationship   = (pd.read_excel(
            io = relationship_path, sheet_name = "Sheet2")
            [["Ticker", "fut_ticker"]].
            rename(columns = {"Ticker": "ticker"}))
        
        alpha_path = os.path.join(self.data_path, "ETFSPYAlpha.parquet")
        df_alpha   = (pd.read_parquet(
            path = alpha_path, engine = "pyarrow")
            [["date", "ticker", "const", "beta"]].
            pivot(index = "date", columns = "ticker", values = "const").
            shift().
            reset_index().
            melt(id_vars = "date", value_name = "lag_eq_alpha").
            dropna())
        
        df_out = (df_alpha.merge(
            right = df_relationship, how = "left", on = ["ticker"]).
            merge(right = df_fut_rtn, how = "left", on = ["date", "fut_ticker"]).
            dropna().
            assign(group_var = lambda x: x.ticker + " " + x.fut_ticker).
            set_index("date").
            groupby("group_var").
            apply(self._oos_resid_calc).
            reset_index())
        
        if verbose: print("Saving Data\n")
        
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _optimize_os_decile(
            self, 
            df     : pd.DataFrame,
            q      : int = 10, 
            min_obs: int = 30) -> pd.DataFrame: 
        
        df      = df.sort_values("date")
        dates   = df.date.drop_duplicates().sort_values().to_list()
        results = []
        
        #for i in range(min_obs, len(dates)):
            
        for i in tqdm(
                iterable = range(min_obs, len(dates)),
                desc     = "OOS Decile"):
            
            date  = dates[i]
            df_is = df.iloc[:i]
            df_os = df.iloc[i:i+1]
            
            if len(df_is) <= min_obs:
                continue
            
            _, bins = pd.qcut(
                x          = df_is["lag_resid"],
                q          = q, 
                retbins    = True,
                duplicates = "drop")
            
            bins[0]  = -np.inf
            bins[-1] = np.inf
            
            df_is_decile = (df_is.assign(
                decile             = lambda x: pd.cut(
                    x              = x.lag_resid,
                    bins           = bins,
                    labels         = range(1, len(bins)),
                    include_lowest = True)))
            
            grp = (df_is_decile.groupby(
                "decile")
                ["fut_rtn"])
            
            sharpe  = (grp.mean() / grp.std()) * np.sqrt(252)
            df_tail = (sharpe.loc[
                sharpe.index.isin([1,2,9,10])].
                to_frame(name = "sharpe"))
            
            if df_tail.empty:
                continue
            
            df_tail["group"] = np.where(df_tail.index <= 2, "lgroup", "ugroup")
            sharpe_prod      = df_tail.groupby("group")["sharpe"].prod()
            
            l_signal = np.where(sharpe_prod.get("lgroup") > 0, 1, 0)
            u_signal = np.where(sharpe_prod.get("ugroup") > 0, 1, 0)
            
            sharpe_dict = sharpe.to_dict()
            last_is     = df_is_decile.iloc[[-1]].copy()
            last_is["signal_scaler"] = (np.select(
                condlist = [
                    last_is["decile"].astype(int) <= 2,
                    last_is["decile"].astype(int) >= 9],
                choicelist = [l_signal, u_signal],
                default    = np.nan))
            
            df_add = (df_os.assign(
                decile        = last_is["decile"].values[0],
                sharpe        = lambda x: x.decile.map(sharpe_dict),
                signal_scaler = last_is["signal_scaler"].values[0],
                signal_rtn    = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fut_rtn))
            
            results.append(df_add)
            
        df_out = (pd.concat(
            objs         = results,
            ignore_index = True))
        
        return df_out
        
    def _optimize_oos_resid(self, verbose: bool = True) -> None:
        
        in_path  = os.path.join(self.data_path, "OOSETFAlphaResid.parquet")
        out_path = os.path.join(self.data_path, "OptimizedOOSDecile.parquet")
        
        if os.path.exists(out_path):
            if verbose: 
                print("Already have Out-of-Sample Optimized Residuals")
                
            return None
        
        if verbose:
            print("Generating Out-of-Sample Optimized Residuals")
        
        df_out  = (pd.read_parquet(
            path = in_path, engine = "pyarrow")
            [["group_var", "date", "fut_rtn", "lag_resid"]].
            dropna().
            groupby("group_var").
            apply(self._optimize_os_decile, include_groups = False).
            reset_index().
            drop(columns = ["level_1"]))
        
        if verbose: 
            print("Saving Data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
            
    oos_decile = OOSDecile()
    #oos_decile._get_etf_alpha()
    #oos_decile._get_oos_resid()
    oos_decile._optimize_oos_resid()
    
if __name__ == "__main__": main()