import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime


class TAcalculator:

    def __init__(self, stock_code: str, start_date: str, end_date: str, column_used: str='Adj Closed'):
        self.stock_code = stock_code
        self.data: pd.DataFrame = yf.download(stock_code, start=start_date, end=end_date, interval='1d')
        self.colname = column_used
        self.strategies: dict = dict()
        self.backtest_details: dict = dict()
        self._backtest_results: dict = dict()
        self.flag_colname = 'Flag'

    @property
    def backtest_results(self):
        return pd.DataFrame(self._backtest_results).T

    def add_strategy(self, strategy_function, *args, **kwargs):
        name, signals = strategy_function(self.data[self.colname], *args, **kwargs)
        valid_signals = self.validate_strategy(signals)
        self.strategies.update({name: valid_signals})

    def backtest_strategies(self):
        # TODO discrete stock number
        for k, v in self.strategies.items():
            temp = self.data.join(v, how='right')
            df_bull = temp[temp[self.flag_colname] == 1].reset_index()
            df_bear = temp[temp[self.flag_colname] == -1].reset_index()

            srs_pct = df_bear[self.colname] / df_bull[self.colname]
            srs_PnL = df_bear[self.colname] - df_bull[self.colname]
            srs_Cap = df_bull[self.colname] - df_bear[self.colname].shift(1).fillna(0)
            
            df_conc = pd.concat({'Bull': df_bull, 'Bear': df_bear},  axis=1)
            df_conc = df_conc.loc[:, (slice(None), ['Date', self.colname, self.flag_colname])]
            df_conc.columns = ['_'.join(e) for e in df_conc.columns.to_flat_index()]

            dict_srs = {'Multiplier': srs_pct, 'PayinCapital': srs_Cap, 'Profit&Loss': srs_PnL}
            df_conc = df_conc.assign(**dict_srs)
            self.backtest_details[k] = df_conc

            dict_val = {'NumOfCycle': df_conc.shape[0], 'Payin_Capital': srs_Cap.sum(), 'Profit_Loss': srs_PnL.sum()}
            dict_val['Result_Asset'] = dict_val['Payin_Capital'] + dict_val['Profit_Loss']
            dict_val['ReturnOnCap'] = dict_val['Result_Asset'] / dict_val['Payin_Capital']
            dict_val['%_Change'] = (srs_pct.product() - 1) * 100
            self._backtest_results[k] = {k: round(v, 4) for k, v in dict_val.items()}

    def validate_strategy(self, strategy):
        if not self.validate_flags_index(strategy):
            strategy = self.ensure_flags_index(strategy)
        if not self.validate_first_bull(strategy):
            strategy = self.ensure_first_bull(strategy)
        if not self.validate_last_bear(strategy):
            strategy = self.ensure_last_bear(strategy)
        return strategy

    def validate_flags_index(self, strategy, verbose=False):
        # strategy must align with data index
        stra_idx = set(strategy.index)
        data_idx = set(self.data.index)
        diff = stra_idx.symmetric_difference(data_idx)
        if len(diff) == 0:
            return True
        elif verbose:
            print(f'Strategy date is not aligned with trading date\n')

    def ensure_flags_index(self, strategy):
        # align index with
        strategy = strategy.copy()
        strategy = strategy.reindex(self.data.index).fillna(0)
        return strategy
    
    def validate_first_bull(self, strategy, verbose=False):
        # strategy must start with 1 "bull', or the first -1 is skipped
        flags = strategy[strategy != 0]
        if flags.iat[0] == 1:
            return True
        elif verbose:
            print(f'First flag is not bull: {flags.index[0]}\n')

    def ensure_first_bull(self, strategy):
        strategy = strategy.copy()
        loc = strategy[strategy != 0].index[0]
        strategy.at[loc] = 0
        return strategy

    def validate_last_bear(self, strategy, verbose=False):
        # strategy must end with -1 "bear:, -1 is assigned at the last date
        flags = strategy[strategy != 0]
        if flags.iat[-1] == -1:
            return True
        elif verbose:
            print(f'Last flag is not bear: {flags.index[-1]}\n')

    def ensure_last_bear(self, strategy):
        strategy = strategy.copy()
        strategy.iat[-1] = -1
        return strategy

    def plot_strategy(self, strategy_name):
        pass
        # Plot moving average with signals and position
        plt.figure(figsize=(20, 10))
        # plot close price, short-term and long-term moving averages
        self.data[self.colname].plot(color='k', label=self.colname)
        # plot 'buy' signals
        flags = self.strategies[strategy_name]
        bull_flags = self.data[self.colname].reindex(flags[flags == 1].index)
        plt.plot(bull_flags.index, bull_flags, '^', markersize=15, color='g', label='buy')
        # plot 'sell' signals
        bear_flags = self.data[self.colname].reindex(flags[flags == -1].index)
        plt.plot(bear_flags.index, bear_flags, 'v', markersize=15, color='r', label='sell')
        plt.ylabel('Price', fontsize=15)
        plt.xlabel('Date', fontsize=15)
        plt.title(self.stock_code + ' at ' + strategy_name, fontsize=20)
        plt.legend()
        plt.grid()
        plt.show()


def ma_strategy(series: pd.Series, short_period: int, long_period: int) -> dict:
    strategy_name = 'sma' + str(short_period).rjust(3, '0') + '>sma' + str(long_period).rjust(3, '0')
    srs_short = series.rolling(window=short_period, min_periods=short_period).mean()
    srs_long = series.rolling(window=long_period, min_periods=long_period).mean()
    diff = np.sign(srs_short - srs_long)
    signal = diff.diff() / 2
    signal.name = 'Flag'
    return strategy_name, signal


class Validator:
    def __init__(self, data_type, scheme_list):
        self.scheme_list = scheme_list

    # validate return true if pass
    def validate(self):
        for e in self.scheme_list:
            pass
