"""
engine package
===============

Core analytical engines for Swing Scanner.

Modules
-------
data
    Market data acquisition and enrichment.

indicators
    Technical indicator calculations.

scanner
    Swing trading signal generation.

backtest
    Strategy simulation and performance analysis.

reports
    Report generation and exports.
"""


###############################################################################
# Public Engine Imports
###############################################################################

from engine.backtest import BacktestEngine
from engine.data import MarketDataEngine
from engine.indicators import IndicatorEngine
from engine.reports import ReportEngine
from engine.scanner import SwingScanner

###############################################################################
# Public API
###############################################################################

__all__ = [

    "MarketDataEngine",

    "IndicatorEngine",

    "SwingScanner",

    "BacktestEngine",

    "ReportEngine",

]