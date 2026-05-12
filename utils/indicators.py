"""Technical indicators calculation"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice

class TechnicalIndicators:
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to dataframe"""
        df = df.copy()
        
        # Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = df['bb_high'] - df['bb_low']
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_change_5d'] = df['close'].pct_change(periods=5)
        
        return df
    
    @staticmethod
    def get_trading_signals(df):
        """Generate buy/sell signals based on indicators"""
        df = df.copy()
        df['signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
        
        # RSI signals
        rsi_buy = df['rsi'] < 30
        rsi_sell = df['rsi'] > 70
        
        # MACD signals
        macd_buy = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        macd_sell = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
        
        # Golden Cross / Death Cross
        golden_cross = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        death_cross = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
        
        # Bollinger Bands signals
        bb_buy = df['close'] < df['bb_low']
        bb_sell = df['close'] > df['bb_high']
        
        # Combine signals
        df.loc[(rsi_buy | macd_buy | golden_cross | bb_buy), 'signal'] = 1
        df.loc[(rsi_sell | macd_sell | death_cross | bb_sell), 'signal'] = -1
        return df