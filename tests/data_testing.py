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
    """
    Convert frequency string to its equivalent DateOffset.

    Args:
        freq (str): Frequency string (e.g. 'D', 'H', 'M', ...).

    Returns:
        DateOffset: Corresponding pandas DateOffset object.

    Raises:
        ValueError: If the provided frequency is unsupported.
    """
    # Define mapping of frequency strings to DateOffset objects.
    mapping = {
        'D': pd.DateOffset(days=1),
        'H': pd.DateOffset(hours=1),
        # 'T' and 'min' are equivalent for minutes
        'T': pd.DateOffset(minutes=1),
        'S': pd.DateOffset(seconds=1),
        # 'L' and 'ms' are equivalent for milliseconds
        'L': pd.DateOffset(milliseconds=1),
        'U': pd.DateOffset(microseconds=1),
        'M': pd.DateOffset(months=1),
        'Y': pd.DateOffset(years=1),
        'Q': pd.DateOffset(months=3),
        # Business day, equivalent to 'W'
        'B': pd.DateOffset(weeks=1),
        'W': pd.DateOffset(weeks=1),
        # Annual frequency, same as 'Y'
        'A': pd.DateOffset(years=1),
    }

    # Check if freq is in the mapping keys. If not, raise an error.
    if freq not in mapping:
        raise ValueError(f"Unsupported frequency: {freq}")

    return mapping[freq]


def _start_of_period(date, freq):
    """
    Roll back the given date to the start of the period based on the given frequency.

    Args:
        date (Timestamp): The input date.
        freq (str): Frequency string (e.g. 'D', 'H', 'M', ...).

    Returns:
        Timestamp: Date adjusted to the start of the period.
    """
    offset = _freq_to_offset(freq)
    rolled_date = date - offset

    # Keep rolling the date back until it reaches the start of the period.
    while (date.to_period(freq) - rolled_date.to_period(freq)).n < 1:
        date = rolled_date
        rolled_date -= offset

    return date


def generate_data(data_type='OHLCV', freq='D', samples=252, ticks_per_sample=10, business_hours=True, start_date=None):
    """
    Generate financial data (OHLCV or tick data) based on input parameters.

    Args:
        data_type (str): Type of data to generate. Options are 'OHLCV' or 'tick'. Default is 'OHLCV'.
        freq (str): Frequency string (e.g. 'D', 'H', 'M', ...). Default is 'D'.
        samples (int): Number of data samples to generate. Default is 252.
        ticks_per_sample (int): Number of tick data points per sample. Used if data_type is 'tick'. Default is 10.
        business_hours (bool): Whether to consider only business hours for generating data. Default is True.
        start_date (Timestamp, optional): Starting date for the data generation. Default is the start of the current year.

    Returns:
        DataFrame: Pandas DataFrame with generated data.

    Raises:
        ValueError: If the provided data_type is neither 'OHLCV' nor 'tick'.
    """
    # If start_date isn't provided, default to the beginning of the current year.
    if start_date is None:
        start_date = pd.Timestamp.now().replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + _freq_to_offset(freq) * samples

    # Generate OHLCV data.
    if data_type == 'OHLCV':
        idx = pd.date_range(end=end_date, periods=samples, freq=freq)
        # If business_hours is True, consider only weekdays.
        if business_hours:
            idx = idx[idx.weekday < 5]

        # Simulate close, open, high, low, and volume data.
        close = 100 + np.cumsum(np.random.randn(len(idx)) * 0.5)
        open_ = close + np.random.randn(len(idx)) * 0.5
        high = close + np.random.rand(len(idx)) * 0.5
        low = close - np.random.rand(len(idx)) * 0.5
        volume = np.random.randint(5000, 10000, size=len(idx))

        # Create the OHLCV DataFrame.
        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=idx)

    # Generate tick data.
    elif data_type == 'tick':
        base_idx = pd.date_range(end=end_date, periods=samples, freq=freq)
        # If business_hours is True, consider only weekdays.
        if business_hours:
            base_idx = base_idx[base_idx.weekday < 5]

        # Simulate ticks for each sample in base_idx.
        all_ticks = []
        for base_time in base_idx:
            # Define market hours if business_hours is True.
            if business_hours:
                start_time = base_time + pd.Timedelta(hours=9, minutes=30)  # Market open
                end_time = base_time + pd.Timedelta(hours=16)  # Market close
            else:
                start_time = base_time
                end_time = base_time + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # Just before the next day starts

            # Simulate tick data points within the defined time range.
            for _ in range(ticks_per_sample):
                tick_time = np.random.uniform(start_time.value, end_time.value)
                tick_time = pd.Timestamp(tick_time)
                all_ticks.append(tick_time)

        idx = pd.DatetimeIndex(sorted(all_ticks))

        # Simulate price and volume data for tick data.
        prices = 100 + np.cumsum(np.random.randn(len(idx)) * 0.1)
        volume = np.random.randint(50, 200, size=len(idx))

        # Create the tick DataFrame.
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
