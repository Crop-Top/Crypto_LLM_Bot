from abc import ABC, abstractmethod

import pandas as pd

from src.trading.models import Signal
from src.trading.predictor import TradePredictor


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        ...


class MLStrategy(BaseStrategy):
    def __init__(self, predictor: TradePredictor):
        self.predictor = predictor

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        signal_str, confidence = self.predictor.predict(data)
        if confidence < 0.50:
            return Signal.HOLD
        return Signal(signal_str)
