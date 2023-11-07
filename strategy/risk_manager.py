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

    def __init__(self, max_drawdown, max_holding_period):
        self.max_drawdown = max_drawdown
        self.max_holding_period = max_holding_period

    @abstractmethod
    def position_size(self, *args, **kwargs):
        pass

class KellyCriterion(RiskManager):

    def __init__(self):
        pass

    def postion_size(self, win_probability, win_loss_ratio):
        """
        Calculate the Kelly Criterion position size.

        :param win_probability: The probability of a winning trade.
        :param win_loss_ratio: The ratio of the average win to the average loss.
        :return: The recommended position size as a fraction of the portfolio.
        """
        # Kelly Criterion formula: f = p - (1 - p) / b
        # where:
        # f is the fraction of the current bankroll to wager;
        # p is the probability of a win;
        # b is the odds received on the wager (b to 1); for money management, we use the win/loss ratio.
        kelly_fraction = win_probability - ((1 - win_probability) / win_loss_ratio)
        return max(0, kelly_fraction)  # Do not bet if the fraction is negative