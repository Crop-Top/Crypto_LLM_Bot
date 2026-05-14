from abc import ABC, abstractmethod

import pandas as pd

from src.trading.models import Signal
from src.trading.predictor import TradePredictor


class BaseStrategy(ABC):
    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> Signal:
        ...


class MLStrategy(BaseStrategy):
    def __init__(self, predictor: TradePredictor, confidence_threshold: float = 0.55):
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold

    def generate_signal(self, data: pd.DataFrame) -> Signal:
        signal_str, confidence = self.predictor.predict(data)
        if confidence < self.confidence_threshold:
            return Signal.HOLD
        return Signal(signal_str)
