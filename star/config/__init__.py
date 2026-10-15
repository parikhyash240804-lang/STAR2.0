"""STAR2.0 configuration: schema, validation and the Excel input template."""
from .schema import (BacktestSettings, CostModel, DataSettings, ForecastSettings, Mandate,
                     ReturnSettings, StarConfig, normalise_tickers)

__all__ = ["BacktestSettings", "CostModel", "DataSettings", "ForecastSettings", "Mandate",
           "ReturnSettings", "StarConfig", "normalise_tickers"]
