"""
MIT License

Copyright (c) 2023 Joseph Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from abc import ABC, abstractmethod

class RiskManager(ABC):

    def __init__(self, max_drawdown = 1, max_holding_period_bars = 100):
        self.max_drawdown = max_drawdown
        self.max_holding_period = max_holding_period_bars

    @abstractmethod
    def calculate(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def stop_loss(self, *args, **kwargs):
        pass

    @abstractmethod
    def take_profit(self, *args, **kwargs):
        pass


class PositionSizer(ABC):

    def __init__(self):
        pass 

    @abstractmethod
    def position_size(self, *args, **kwargs):
        pass


class KellyCriterion(PositionSizer):

    def __init__(self, win_probability = 0.5, win_loss_ratio = 0.5):
        super().__init__()
        self.win_probability = win_probability
        self.win_loss_ratio = win_loss_ratio

    def update_probs(self, win_probability, win_loss_ratio):
        self.win_probability = win_probability
        self.win_loss_ratio = win_loss_ratio

    def postion_size(self):
        """
        Calculate the Kelly Criterion position size.

        :return: The recommended position size as a fraction of the portfolio.
        """
        # Kelly Criterion formula: f = p - (1 - p) / b
        # where:
        # f is the fraction of the current bankroll to wager
        # p is the probability of a win
        # b is the odds received on the wager (b to 1); for money management, we use the win/loss ratio.
        kelly_fraction = self.win_probability - ((1 - self.win_probability) / self.win_loss_ratio)
        return max(0, kelly_fraction)  # Do not bet if the fraction is negative
    

class FixedPercent(PositionSizer):

    def __init__(self, percent):
        super().__init__()
        self.percent = percent

    def position_size(self, price):
        return price 

