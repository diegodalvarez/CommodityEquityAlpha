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
tqdm.pandas()

class OOSDecile:
    
    def __init__(self) -> None: 
        
        self.cur_path  = os.getcwd()
        self.repo_path = os.path.abspath(os.path.join(self.cur_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        
        self.zscore_window = 30
        self.min_nobs      = 30
        self.q             = 10
        self.train_sizes   = [0.3, 0.5, 0.7]
        
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
            
    def _oos_resid_calc(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        '''
        For computing the Expanding Out-of-Sample Residuals
        '''
        
        df_out = (RollingOLS(
            endog     = df.fut_rtn,
            exog      = sm.add_constant(df.lag_eq_alpha),
            expanding = True,
            min_nobs  = self.min_nobs).
            fit().
            params.
            rename(columns = {"lag_eq_alpha": "lag_alpha_beta"}).
            merge(right = df, how = "inner", on = ["date"]).
            assign(
                y_pred    = lambda x: (x.lag_alpha_beta * x.lag_eq_alpha) + x.const,
                resid     = lambda x: x.fut_rtn - x.y_pred,
                lag_resid = lambda x: x.resid.shift(),
                zscore   = lambda x: (
                    x.resid - x.resid.ewm(span = self.zscore_window, adjust = False).mean()) /
                    x.resid.ewm(span = self.zscore_window, adjust = False).std(),
                lag_zscore = lambda x: x.zscore.shift()))
    
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
        
        tmp_path = os.path.join(self.data_path, "InSampleOLSSetup.parquet")
        df_out   = (pd.read_parquet(
            path = tmp_path, engine = "pyarrow").
            query("rtn_calc == 'raw_rtn'").
            query("group == 'zscore'").
            drop(columns = ["rtn_calc", "group"]).
            assign(group_var = lambda x: x.etf_ticker + " " + x.fut_ticker).
            set_index("date").
            groupby("group_var").
            apply(self._oos_resid_calc).
            reset_index().
            drop(columns = ["group_var"]))
        
        '''
        return-1
        
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
        '''
        
        if verbose: print("Saving Data\n")
        
        df_out.to_parquet(path = out_path, engine = "pyarrow")
    
    def _walk_forward_optimize_os_decile(
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
                x          = df_is["value"],
                q          = q, 
                retbins    = True,
                duplicates = "drop")
            
            bins[0]  = -np.inf
            bins[-1] = np.inf
            
            df_is_decile = (df_is.assign(
                decile             = lambda x: pd.cut(
                    x              = x.value,
                    bins           = bins,
                    labels         = range(1, len(bins)),
                    include_lowest = True).
                shift()))
            
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
        
    def _walk_forward_optimize_oos_resid(self, verbose: bool = True) -> None:
        
        in_path  = os.path.join(self.data_path, "OOSETFAlphaResid.parquet")
        out_path = os.path.join(self.data_path, "OptimizedOOSDecile.parquet")
        
        if os.path.exists(out_path):
            if verbose: 
                print("Already have Out-of-Sample Optimized Residuals")
                
            return None
        
        if verbose:
            print("Generating Out-of-Sample Optimized Residuals")
            
        df_prep = (pd.read_parquet(
            path = in_path, engine = "pyarrow")
            [[
                "date", "etf_ticker", "fut_ticker", "resid", 
                "zscore", "fut_rtn"]].
            melt(id_vars = ["date", "etf_ticker", "fut_ticker", "fut_rtn"]).
            dropna().
            assign(group_var = lambda x: x.etf_ticker + " " + x.fut_ticker + " " + x.variable))
        
        df_out = (df_prep.groupby(
            "group_var").
            apply(self._optimize_os_decile, include_groups = False).
            reset_index().
            drop(columns = ["group_var", "level_1"]))
        
        '''
        display(df_out.to_parquet(path = "tmp.parquet"))
        return-1
        
        df_out  = (pd.read_parquet(
            path = in_path, engine = "pyarrow")
            [["group_var", "date", "fut_rtn", "lag_resid"]].
            dropna().
            groupby("group_var").
            apply(self._optimize_os_decile, include_groups = False).
            reset_index().
            drop(columns = ["level_1"]))
        '''
        
        if verbose: 
            print("Saving Data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _train_test_split_decile_optimize(self, df: pd.DataFrame, sample_size: float) -> pd.DataFrame: 
        
        slice_date = df["date"].quantile(sample_size)
        df_is  = df[df["date"] <= slice_date]
        df_oos = df[df["date"] >  slice_date]
        
        _, bins  = pd.qcut(x = df_is.value, q = self.q, retbins = True)
        bins[0]  = -np.inf
        bins[-1] = np.inf
    
        df_is_decile = (df_is.assign(
            sample_group = "in_sample",
            decile       = lambda x: pd.cut(
                x      = x.value,
                bins   = bins,
                labels = range(1, len(bins))).
                shift()).
            dropna())
    
        df_out_decile = (df_oos.assign(
            sample_group = "out_sample",
            decile       = lambda x: pd.cut(
                x      = x.value,
                bins   = bins,
                labels = range(1, len(bins))).
            shift()).
            dropna())
    
        df_decile_sharpe = (df_is_decile[
            ["decile", "fut_rtn"]].
            groupby("decile").
            agg(lambda x: x.mean() / x.std() * np.sqrt(252)).
            rename(columns = {"fut_rtn": "sharpe"}))
    
        df_decile_tmp = (df_decile_sharpe.reset_index().query(
            "decile == [1,2,9,10]").
            assign(decile_group = lambda x: np.where(x.decile <= 2, "lgroup", "ugroup")))
    
        df_signal_scaler = (df_decile_tmp.drop(
            columns = ["decile"]).
            groupby("decile_group").
            agg("prod").
            assign(signal_scaler = lambda x: np.where(x.sharpe > 0, 1, np.nan)).
            drop(columns = ["sharpe"]).
            reset_index().
            merge(right = df_decile_tmp, how = "outer", on = ["decile_group"]))
    
        df_out = (pd.concat([
            df_is_decile, df_out_decile]).
            merge(right = df_signal_scaler, how = "outer", on = ["decile"]).
            assign(
                slice_date = slice_date,
                signal_rtn = lambda x: np.sign(x.signal_scaler * x.sharpe) * x.fut_rtn))
    
        return df_out

        
    def _train_test_split_oos_optimization(self, verbose: bool = True) -> None: 
        
        in_path  = os.path.join(self.data_path, "OOSETFAlphaResid.parquet")
        out_path = os.path.join(self.data_path, "TrainTestOosOptimizedResiduals.parquet")
        
        if os.path.exists(out_path):
            if verbose: 
                print("Already have Train/Test Split Out-of-Sample Optimized Residuals")
                
            return None
        
        if verbose:
            print("Generating Train/Test Split Out-of-Sample Optimized Residuals")
            
        df_prep = (pd.read_parquet(
            path = in_path, engine = "pyarrow")
            [[
                "date", "etf_ticker", "fut_ticker", "resid", 
                "zscore", "fut_rtn"]].
            melt(id_vars = ["date", "etf_ticker", "fut_ticker", "fut_rtn"]).
            dropna().
            assign(group_var = lambda x: x.etf_ticker + " " + x.fut_ticker + " " + x.variable).
            dropna())
        
        df_lists = []
            
        for train_size in self.train_sizes:
            
            if verbose: print("Working on {} sample size".format(train_size))
            df_add = (pd.read_parquet(
                path = in_path, engine = "pyarrow")
                [["date", "etf_ticker", "fut_ticker", "fut_rtn", "resid", "zscore"]].
                dropna().
                melt(id_vars = ["date", "etf_ticker", "fut_ticker", "fut_rtn"]).
                assign(name = lambda x: x.etf_ticker + " " + x.fut_ticker + " " + x.variable).
                groupby("name").
                #apply(self._train_test_split_decile_optimize, train_size).
                progress_apply(lambda group: self._train_test_split_decile_optimize(group, train_size)).
                reset_index().
                drop(columns = ["level_1"]).
                assign(sample_size = train_size))
    
            print(" ")
            
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        

def main() -> None: 
            
    oos_decile = OOSDecile()
    #oos_decile._get_etf_alpha()
    #oos_decile._get_oos_resid()
    oos_decile._optimize_oos_resid()
    
#if __name__ == "__main__": main()

oos_decile = OOSDecile()
#oos_decile._get_oos_resid()
#oos_decile._walk_forward_optimize_oos_resid()
df = oos_decile._train_test_split_oos_optimization()