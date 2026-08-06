"""
engine/indicators.py
====================

Institutional Technical Indicator Engine.

Responsibilities
----------------
* Validate OHLCV market data
* Calculate reusable technical indicators
* Return an enriched market DataFrame

Notes
-----
This module contains NO trading logic.

It does NOT:
    * Generate BUY / SELL signals
    * Rank stocks
    * Select trades
    * Backtest strategies

Those responsibilities belong to scanner.py.
"""

from __future__ import annotations

###############################################################################
# Standard Library
###############################################################################
import logging
from typing import Any

###############################################################################
# Third Party
###############################################################################
import pandas as pd

try:

    import talib

    HAS_TALIB = True

except ImportError:

    HAS_TALIB = False

import pandas_ta as ta

###############################################################################
# Local Imports
###############################################################################
from config import settings

###############################################################################
# Logger
###############################################################################

logger = logging.getLogger(__name__)

###############################################################################
# Indicator Engine
###############################################################################


class IndicatorEngine:
    """
    Institutional Technical Indicator Engine.

    Pipeline
    --------

        Validate

            ↓

        Trend

            ↓

        Momentum

            ↓

        Volume

            ↓

        Volatility

            ↓

        Market Structure

            ↓

        Return DataFrame
    """

    REQUIRED_COLUMNS = (

        "Date",

        "Symbol",

        "Open",

        "High",

        "Low",

        "Close",

        "Volume",

    )

    ###########################################################################
    # Initialization
    ###########################################################################

    def __init__(self) -> None:
        """Initialize indicator engine."""

        self.settings = settings

        self.backend = (
            "talib"
            if HAS_TALIB
            else "pandas_ta"
        )

        logger.info(

            "IndicatorEngine initialized "

            "(backend=%s).",

            self.backend,

        )

    ###########################################################################
    # Validation
    ###########################################################################

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Validate OHLCV dataframe.
        """

        self._validate_dataframe(
            dataframe,
        )

        self._validate_columns(
            dataframe,
        )

        dataframe = dataframe.copy()

        dataframe = dataframe.sort_values(
            [
                "Symbol",
                "Date",
            ]
        )

        dataframe = dataframe.drop_duplicates(
            subset=[
                "Symbol",
                "Date",
            ]
        )

        dataframe = dataframe.reset_index(
            drop=True,
        )

        return dataframe

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate dataframe object.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Expected pandas DataFrame."
            )

        if dataframe.empty:

            raise ValueError(
                "Market dataframe is empty."
            )

    def _validate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate required columns.
        """

        missing = [

            column

            for column in self.REQUIRED_COLUMNS

            if column not in dataframe.columns

        ]

        if missing:

            raise ValueError(

                "Missing columns: "

                f"{missing}"

            )

    ###########################################################################
    # Pipeline
    ###########################################################################

    def calculate_all(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate all supported indicators.
        """

        dataframe = self.validate(
            dataframe,
        )

        dataframe = self.calculate_trend(
            dataframe,
        )

        dataframe = self.calculate_momentum(
            dataframe,
        )

        dataframe = self.calculate_volume(
            dataframe,
        )

        dataframe = self.calculate_volatility(
            dataframe,
        )

        dataframe = self.calculate_market_structure(
            dataframe,
        )

        dataframe = self.validate_output(
            dataframe,
        )

        logger.info(
            "Indicator calculation complete."
        )

        return dataframe

    ###############################################################################
    # Trend Indicators
    ###############################################################################

    def calculate_trend(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate trend indicators.
        """

        logger.info("Calculating trend indicators.")

        dataframe = self._ema(dataframe)
        dataframe = self._sma(dataframe)
        dataframe = self._adx(dataframe)
        dataframe = self._supertrend(dataframe)

        return dataframe


    def _ema(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate EMA 20/50/100/200.
        """

        periods = (20, 50, 100, 200)

        for period in periods:

            column = f"EMA_{period}"

            if HAS_TALIB:

                dataframe[column] = (

                    dataframe
                    .groupby("Symbol")["Close"]
                    .transform(
                        lambda x: talib.EMA(
                            x.to_numpy(),
                            timeperiod=period,
                        )
                    )

                )

            else:

                dataframe[column] = (

                    dataframe
                    .groupby("Symbol")["Close"]
                    .transform(
                        lambda x, p=period: ta.ema(
                            x,
                            length=period,
                        )
                    )

                )

        return dataframe


    def _sma(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate SMA.
        """

        periods = (20, 50, 200)

        for period in periods:

            column = f"SMA_{period}"

            if HAS_TALIB:

                dataframe[column] = (

                    dataframe
                    .groupby("Symbol")["Close"]
                    .transform(
                        lambda x, p=period: talib.SMA(
                            x.to_numpy(),
                            timeperiod=period,
                        )
                    )

                )

            else:

                dataframe[column] = (

                    dataframe
                    .groupby("Symbol")["Close"]
                    .transform(
                        lambda x: ta.sma(
                            x,
                            length=period,
                        )
                    )

                )

        return dataframe


    def _adx(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Average Directional Index.
        """

        def calculate(group):

            if HAS_TALIB:

                group["ADX_14"] = talib.ADX(

                    group["High"],

                    group["Low"],

                    group["Close"],

                    timeperiod=14,

                )

            else:

                result = ta.adx(

                    high=group["High"],

                    low=group["Low"],

                    close=group["Close"],

                    length=14,

                )

                group["ADX_14"] = result.iloc[:, 0]

            return group

        return dataframe.groupby(
            "Symbol",
            group_keys=False,
        ).apply(calculate)


    def _supertrend(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Supertrend.
        """

        def calculate(group):

            result = ta.supertrend(

                high=group["High"],

                low=group["Low"],

                close=group["Close"],

                length=10,

                multiplier=3,

            )

            group["SUPERTREND"] = result.iloc[:, 0]

            group["SUPERTREND_DIR"] = result.iloc[:, 1]

            return group

        return dataframe.groupby(
            "Symbol",
            group_keys=False,
        ).apply(calculate)


    ###############################################################################
    # Momentum Indicators
    ###############################################################################

    def calculate_momentum(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate momentum indicators.
        """

        logger.info("Calculating momentum indicators.")

        dataframe = self._rsi(dataframe)
        dataframe = self._macd(dataframe)
        dataframe = self._roc(dataframe)
        dataframe = self._ppo(dataframe)
        dataframe = self._stoch_rsi(dataframe)

        return dataframe


    def _rsi(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        RSI(14).
        """

        if HAS_TALIB:

            dataframe["RSI_14"] = (

                dataframe
                .groupby("Symbol")["Close"]
                .transform(
                    lambda x: talib.RSI(
                        x.to_numpy(),
                        timeperiod=14,
                    )
                )

            )

        else:

            dataframe["RSI_14"] = (

                dataframe
                .groupby("Symbol")["Close"]
                .transform(
                    lambda x: ta.rsi(
                        x,
                        length=14,
                    )
                )

            )

        return dataframe


    def _macd(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        MACD.
        """

        def calculate(group):

            if HAS_TALIB:

                macd, signal, hist = talib.MACD(
                    group["Close"],
                )

                group["MACD"] = macd
                group["MACD_SIGNAL"] = signal
                group["MACD_HIST"] = hist

            else:

                result = ta.macd(
                    group["Close"]
                )

                group["MACD"] = result.iloc[:, 0]
                group["MACD_SIGNAL"] = result.iloc[:, 1]
                group["MACD_HIST"] = result.iloc[:, 2]

            return group

        return dataframe.groupby(
            "Symbol",
            group_keys=False,
        ).apply(calculate)


    def _roc(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rate of Change.
        """

        dataframe["ROC_20"] = (

            dataframe
            .groupby("Symbol")["Close"]
            .pct_change(20)
            * 100

        )

        return dataframe


    def _ppo(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Percentage Price Oscillator.
        """

        dataframe["PPO"] = (

            dataframe
            .groupby("Symbol")["Close"]
            .transform(
                lambda x: ta.ppo(x)
            )

        )

        return dataframe


    def _stoch_rsi(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Stochastic RSI.
        """

        def calculate(group):

            result = ta.stochrsi(
                group["Close"]
            )

            group["STOCH_RSI_K"] = result.iloc[:, 0]
            group["STOCH_RSI_D"] = result.iloc[:, 1]

            return group

        return dataframe.groupby(
            "Symbol",
            group_keys=False,
        ).apply(calculate)

    ###############################################################################
    # Volume Indicators
    ###############################################################################

    def calculate_volume(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate volume indicators.
        """

        logger.info(
            "Calculating volume indicators."
        )

        dataframe = self._volume_sma(dataframe)
        dataframe = self._relative_volume(dataframe)
        dataframe = self._obv(dataframe)
        dataframe = self._cmf(dataframe)
        dataframe = self._mfi(dataframe)
        dataframe = self._vwap(dataframe)

        return dataframe


    def _volume_sma(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Volume SMA(20).
        """

        dataframe["VOL_SMA_20"] = (

            dataframe
            .groupby("Symbol")["Volume"]
            .transform(
                lambda x: x.rolling(
                    20,
                    min_periods=1,
                ).mean()
            )

        )

        return dataframe


    def _relative_volume(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Relative Volume.
        """

        dataframe["RVOL_20"] = (

            dataframe["Volume"]

            / dataframe["VOL_SMA_20"]

        )

        return dataframe


    def _obv(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        On Balance Volume.
        """

        def calculate(group):

            if HAS_TALIB:

                group["OBV"] = talib.OBV(

                    group["Close"],

                    group["Volume"],

                )

            else:

                group["OBV"] = ta.obv(

                    close=group["Close"],

                    volume=group["Volume"],

                )

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)


    def _cmf(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Chaikin Money Flow.
        """

        def calculate(group):

            group["CMF_20"] = ta.cmf(

                high=group["High"],

                low=group["Low"],

                close=group["Close"],

                volume=group["Volume"],

                length=20,

            )

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)


    def _mfi(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Money Flow Index.
        """

        def calculate(group):

            if HAS_TALIB:

                group["MFI_14"] = talib.MFI(

                    group["High"],

                    group["Low"],

                    group["Close"],

                    group["Volume"],

                    timeperiod=14,

                )

            else:

                group["MFI_14"] = ta.mfi(

                    high=group["High"],

                    low=group["Low"],

                    close=group["Close"],

                    volume=group["Volume"],

                    length=14,

                )

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)


    def _vwap(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        VWAP.
        """

        def calculate(group):

            group["VWAP"] = ta.vwap(

                high=group["High"],

                low=group["Low"],

                close=group["Close"],

                volume=group["Volume"],

            )

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)


    ###############################################################################
    # Volatility Indicators
    ###############################################################################

    def calculate_volatility(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate volatility indicators.
        """

        logger.info(
            "Calculating volatility indicators."
        )

        dataframe = self._atr(dataframe)
        dataframe = self._bollinger(dataframe)
        dataframe = self._donchian(dataframe)
        dataframe = self._keltner(dataframe)

        return dataframe


    def _atr(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Average True Range.
        """

        def calculate(group):

            if HAS_TALIB:

                atr = talib.ATR(

                    group["High"],

                    group["Low"],

                    group["Close"],

                    timeperiod=14,

                )

            else:

                atr = ta.atr(

                    high=group["High"],

                    low=group["Low"],

                    close=group["Close"],

                    length=14,

                )

            group["ATR_14"] = atr

            group["ATR_PCT"] = (

                atr

                / group["Close"]

                * 100

            )

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)


    def _bollinger(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Bollinger Bands.
        """

        def calculate(group):

            bb = ta.bbands(

                group["Close"],

                length=20,

            )

            group["BB_UPPER"] = bb.iloc[:, 0]

            group["BB_MIDDLE"] = bb.iloc[:, 1]

            group["BB_LOWER"] = bb.iloc[:, 2]

            group["BB_WIDTH"] = (

                group["BB_UPPER"]

                - group["BB_LOWER"]

            ) / group["BB_MIDDLE"]

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)


    def _donchian(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Donchian Channel.
        """

        dataframe["DONCHIAN_HIGH"] = (

            dataframe
            .groupby("Symbol")["High"]
            .transform(
                lambda x: x.rolling(
                    20,
                    min_periods=1,
                ).max()
            )

        )

        dataframe["DONCHIAN_LOW"] = (

            dataframe
            .groupby("Symbol")["Low"]
            .transform(
                lambda x: x.rolling(
                    20,
                    min_periods=1,
                ).min()
            )

        )

        return dataframe


    def _keltner(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Keltner Channel.
        """

        def calculate(group):

            kc = ta.kc(

                high=group["High"],

                low=group["Low"],

                close=group["Close"],

            )

            group["KC_UPPER"] = kc.iloc[:, 0]

            group["KC_MIDDLE"] = kc.iloc[:, 1]

            group["KC_LOWER"] = kc.iloc[:, 2]

            return group

        return dataframe.groupby(

            "Symbol",

            group_keys=False,

        ).apply(calculate)

    ###############################################################################
    # Market Structure
    ###############################################################################

    def calculate_market_structure(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate market structure indicators.
        """

        logger.info(
            "Calculating market structure."
        )

        dataframe = self._rolling_extremes(
            dataframe
        )

        dataframe = self._distance_from_extremes(
            dataframe
        )

        dataframe = self._gap_analysis(
            dataframe
        )

        dataframe = self._average_range(
            dataframe
        )

        dataframe = self._average_body(
            dataframe
        )

        return dataframe


    def _rolling_extremes(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate rolling 52-week high and low.
        """

        dataframe["HIGH_252"] = (

            dataframe
            .groupby("Symbol")["High"]
            .transform(
                lambda x: x.rolling(
                    252,
                    min_periods=1,
                ).max()
            )

        )

        dataframe["LOW_252"] = (

            dataframe
            .groupby("Symbol")["Low"]
            .transform(
                lambda x: x.rolling(
                    252,
                    min_periods=1,
                ).min()
            )

        )

        return dataframe


    def _distance_from_extremes(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate distance from 52-week high and low.
        """

        dataframe["DIST_HIGH_PCT"] = (

            (
                dataframe["Close"]
                - dataframe["HIGH_252"]
            )

            / dataframe["HIGH_252"]

            * 100

        )

        dataframe["DIST_LOW_PCT"] = (

            (
                dataframe["Close"]
                - dataframe["LOW_252"]
            )

            / dataframe["LOW_252"]

            * 100

        )

        return dataframe


    def _gap_analysis(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate overnight gap percentage.
        """

        previous_close = (

            dataframe
            .groupby("Symbol")["Close"]
            .shift(1)

        )

        dataframe["GAP_PCT"] = (

            (
                dataframe["Open"]
                - previous_close
            )

            / previous_close

            * 100

        )

        return dataframe


    def _average_range(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate average trading range.
        """

        dataframe["DAILY_RANGE"] = (

            dataframe["High"]

            - dataframe["Low"]

        )

        dataframe["AVG_RANGE_20"] = (

            dataframe
            .groupby("Symbol")["DAILY_RANGE"]
            .transform(
                lambda x: x.rolling(
                    20,
                    min_periods=1,
                ).mean()
            )

        )

        return dataframe


    def _average_body(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate average candle body.
        """

        dataframe["BODY_SIZE"] = (

            dataframe["Close"]

            - dataframe["Open"]

        ).abs()

        dataframe["AVG_BODY_20"] = (

            dataframe
            .groupby("Symbol")["BODY_SIZE"]
            .transform(
                lambda x: x.rolling(
                    20,
                    min_periods=1,
                ).mean()
            )

        )

        return dataframe


    ###############################################################################
    # Output Validation
    ###############################################################################

    def validate_output(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Final validation before returning.
        """

        dataframe = dataframe.copy()

        dataframe = dataframe.drop_duplicates(
            subset=[
                "Symbol",
                "Date",
            ]
        )

        dataframe = dataframe.sort_values(
            [
                "Symbol",
                "Date",
            ]
        )

        dataframe = dataframe.reset_index(
            drop=True,
        )

        return dataframe


    ###########################################################################
    # Summary
    ###########################################################################

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Indicator engine summary.
        """

        return {

            "engine": "IndicatorEngine",
            "backend": self.backend,
            "trend": True,
            "momentum": True,
            "volume": True,
            "volatility": True,
            "market_structure": True,
        }