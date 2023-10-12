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
import pandas as pd
import numpy as np
import unittest

from data_management.data_manager import PricingSeries


def _freq_to_offset(freq):
    """Map frequency string to its equivalent DateOffset."""
    mapping = {
        'D': pd.DateOffset(days=1),
        'H': pd.DateOffset(hours=1),
        'T': pd.DateOffset(minutes=1),  # or 'min'
        'S': pd.DateOffset(seconds=1),
        'L': pd.DateOffset(milliseconds=1),  # or 'ms'
        'U': pd.DateOffset(microseconds=1),
        'M': pd.DateOffset(months=1),
        'Y': pd.DateOffset(years=1),
        'Q': pd.DateOffset(months=3),
        'B': pd.DateOffset(weeks=1),  # Business day, it's similar to 'W'
        'W': pd.DateOffset(weeks=1),
        'A': pd.DateOffset(years=1),  # Annual frequency, same as 'Y'
    }

    if freq not in mapping:
        raise ValueError(f"Unsupported frequency: {freq}")

    return mapping[freq]


def _start_of_period(date, freq):
    """Roll back the given date to the start of the period based on the given frequency."""
    offset = _freq_to_offset(freq)
    rolled_date = date - offset

    while (date.to_period(freq) - rolled_date.to_period(freq)).n < 1:
        date = rolled_date
        rolled_date -= offset

    return date


def generate_data(data_type='OHLCV', freq='D', samples=252, ticks_per_sample=10, business_hours=True, start_date=None):
    # Adjust the end_date to the start of the period based on frequency.
    if start_date is None:
        start_date = pd.Timestamp.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + _freq_to_offset(freq) * samples

    if data_type == 'OHLCV':
        idx = pd.date_range(end=end_date, periods=samples, freq=freq)
        if business_hours:
            idx = idx[idx.weekday < 5]  # Only consider weekdays for business hours

        close = 100 + np.cumsum(np.random.randn(len(idx)) * 0.5)
        open_ = close + np.random.randn(len(idx)) * 0.5
        high = close + np.random.rand(len(idx)) * 0.5
        low = close - np.random.rand(len(idx)) * 0.5
        volume = np.random.randint(5000, 10000, size=len(idx))

        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=idx)

    elif data_type == 'tick':
        base_idx = pd.date_range(end=end_date, periods=samples, freq=freq)
        if business_hours:
            base_idx = base_idx[base_idx.weekday < 5]  # Only consider weekdays for business hours

        all_ticks = []
        for base_time in base_idx:
            if business_hours:
                start_time = base_time + pd.Timedelta(hours=9, minutes=30)  # Market open
                end_time = base_time + pd.Timedelta(hours=16)  # Market close
            else:
                start_time = base_time
                end_time = base_time + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Just before the next day starts

            for _ in range(ticks_per_sample):
                tick_time = np.random.uniform(start_time.value, end_time.value)
                tick_time = pd.Timestamp(tick_time)
                all_ticks.append(tick_time)

        idx = pd.DatetimeIndex(sorted(all_ticks))

        prices = 100 + np.cumsum(np.random.randn(len(idx)) * 0.1)
        volume = np.random.randint(50, 200, size=len(idx))

        df = pd.DataFrame({
            'price': prices,
            'volume': volume
        }, index=idx)

    else:
        raise ValueError("data_type must be either 'OHLCV' or 'tick'")

    return df


class TestPricingSeries(unittest.TestCase):

    def test_tick_data_loading(self):
        # Generate tick data
        df = generate_data(data_type='tick', samples=1, ticks_per_sample=1000, business_hours=False)
        ps = PricingSeries(df, is_tick_data=True, timeframes=['T', '5T', '15T'])

        self.assertIsInstance(ps.raw_data, pd.DataFrame)
        self.assertTrue(ps.is_tick_data)
        self.assertEqual(ps.base_timeframe, "TICK")
        self.assertIn('5T', ps.timeframes)

    def test_OHLCV_data_loading(self):
        # Generate OHLCV data
        df = generate_data(data_type='OHLCV', freq='T', samples=50, business_hours=False)
        ps = PricingSeries(df, timeframes=['T', '5T', '15T'])

        self.assertIsInstance(ps.raw_data, pd.DataFrame)
        self.assertFalse(ps.is_tick_data)
        self.assertEqual(ps.base_timeframe, "T")
        self.assertIn('5T', ps.timeframes)

    def test_resampling(self):
        # Test resampling with OHLCV data
        df = generate_data(data_type='OHLCV', freq='T', samples=50, business_hours=False)
        ps = PricingSeries(df, timeframes=['T', '5T', '15T'])

        self.assertIn('5T', ps.data)
        self.assertEqual(len(ps.data['5T']), len(df) // 5)

    def test_iteration(self):
        df = generate_data(data_type='tick', samples=1, ticks_per_sample=1000, business_hours=False)
        ps = PricingSeries(df, is_tick_data=True, timeframes=['T', '5T', '15T'])

        for idx, data in ps:
            self.assertIn('T', data)
            self.assertIn('5T', data)
            self.assertIn('15T', data)

    def test_get(self):
        df = generate_data(data_type='OHLCV', freq='T', samples=50, business_hours=False)
        ps = PricingSeries(df, timeframes=['T', '5T', '15T'])

        current_close = ps.get('close', 'T')
        past_close = ps.get('close', 'T', n=1)
        future_close = ps.get('close', 'T', n=1, future=True)

        self.assertNotEqual(current_close, past_close)
        self.assertNotEqual(current_close, future_close)
        self.assertNotEqual(past_close, future_close)


if __name__ == '__main__':
    unittest.main()
