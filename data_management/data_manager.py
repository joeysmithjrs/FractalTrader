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
import sqlite3


def _load_data(data_source, table_name=None):
    """
    Load data from various sources into a DataFrame.

    Args:
        data_source: Either a file path string, a DataFrame, a numpy array, or a list of lists.
        table_name (optional): Required if data_source is a SQLite database file.

    Returns:
        DataFrame with loaded data.

    Raises:
        ValueError: If an unsupported file format is provided or if there are issues with the data format.
    """
    if isinstance(data_source, str):
        # Load data from a file
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source, parse_dates=True, index_col=0)
        elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
            df = pd.read_excel(data_source, parse_dates=True, index_col=0)
        elif data_source.endswith('.db'):
            # Connect to SQLite database and fetch data
            conn = sqlite3.connect(data_source)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=True, index_col=0)
            conn.close()
        else:
            raise ValueError("Unsupported file format!")
    elif isinstance(data_source, pd.DataFrame):
        # Ensure first column is a datetime index
        df = data_source.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
                df.drop(df.columns[0], axis=1, inplace=True)
            except:
                raise ValueError("Error converting the first column to datetime!")
    elif isinstance(data_source, np.ndarray):
        # Convert numpy array to DataFrame and ensure first column is a datetime index
        df = pd.DataFrame(data_source)
        try:
            df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
            df.drop(df.columns[0], axis=1, inplace=True)
        except:
            raise ValueError("Error converting the first column to datetime!")
    elif isinstance(data_source, list):
        # Convert list of lists to DataFrame and ensure first column is a datetime index
        if all(isinstance(i, list) for i in data_source):
            df = pd.DataFrame(data_source)
            try:
                df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
                df.drop(df.columns[0], axis=1, inplace=True)
            except:
                raise ValueError("Error converting the first column to datetime!")
        else:
            raise ValueError("If data_source is a list, it should be a list of lists!")
    else:
        raise ValueError("data_source should be a path to a csv/excel/sqlite database, a pandas DataFrame, "
                         "a numpy array, or a list of lists!")
    return df


def _ohclv_tick_cols(df):
    """
    Rename columns based on their type (OHLCV or TICK).

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with renamed columns.
    """
    ohlcv_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    tick_cols = ["price", "volume", "open_interest"]

    # If there are 3 columns or fewer, assume it's tick data; otherwise, OHLCV
    if df.shape[1] <= 3:
        df.columns = tick_cols[:df.shape[1]]
    else:
        df.columns = ohlcv_cols[:df.shape[1]]
    return df


def _freq_to_timedelta(freq):
    """
    Convert a frequency string to its equivalent timedelta or offset.

    Args:
        freq: Frequency string e.g., '5T', 'D', 'M'.

    Returns:
        Corresponding timedelta or offset object.

    Raises:
        ValueError: If the frequency is not directly convertible to a timedelta.
    """
    if freq == "TICK":
        # Defaults to 1 microsecond because tick data has no consistent time delta
        return pd.to_timedelta("1us")

    # If freq doesn't start with a number and isn't a special case, prepend with '1'
    if not freq[0].isdigit() and freq not in ["M", "Y", "Q", "B"]:
        freq = '1' + freq

    # Default conversion using pandas to_timedelta
    try:
        return pd.to_timedelta(freq)
    except ValueError:
        # Handle special cases for non-fixed duration timeframes
        if 'M' in freq:
            return pd.offsets.MonthEnd()
        elif 'Y' in freq:
            return pd.offsets.YearEnd()
        elif 'Q' in freq:
            return pd.offsets.QuarterEnd()
        elif 'B' in freq:
            return pd.offsets.BDay()
        else:
            raise ValueError(f"The timeframe '{freq}' is not directly convertible to a timedelta.")


class PricingSeries:
    """
    A class that represents a series of pricing data for one or multiple timeframes.

    Attributes:
        raw_data: The raw pricing data.
        symbol: Symbol of the asset.
        is_tick_data: Boolean indicating if data is tick data.
        base_timeframe: The inferred base timeframe from the raw data.
        timeframes: List of all timeframes used.
        data: Resampled data for different timeframes.
        current_index: Index pointer for iteration purposes.
    """

    def __init__(self, data_source, symbol=None, timeframes=None, is_tick_data=False, table_name=None):
        """
        Initialize the PricingSeries object with raw data, symbol, timeframes, etc.

        Args:
            data_source: Data source can be path to file, DataFrame, numpy array or list.
            symbol: Symbol of the asset.
            timeframes: List of timeframes.
            is_tick_data: If the provided data is tick data.
            table_name: Name of the table in case data_source is a database.
        """
        self.raw_data = _ohclv_tick_cols(_load_data(data_source, table_name))
        self.symbol = symbol
        self.is_tick_data = is_tick_data
        self.base_timeframe = self._infer_frequency(self.raw_data.index)
        self.base_delta = _freq_to_timedelta(self.base_timeframe)
        timeframes_td = [_freq_to_timedelta(tf) for tf in timeframes or []]
        self.timeframes = [self.base_timeframe] + [tf for tf, tf_td in zip(timeframes or [], timeframes_td) if
                                                   tf_td >= self.base_delta and tf != self.base_timeframe]
        self.data = self._resample_data()
        self.current_row, self.current_date = None, None
        self.current_index = -1

    def _resample_data(self):
        """
        Resample the raw data to create data for all specified timeframes.

        Returns:
            Dictionary with keys as timeframes and values as respective resampled data.
        """
        resampled = {}

        for tf in self.timeframes:
            if tf == self.base_timeframe:
                resampled[tf] = self.raw_data.copy()
                continue
            if self.is_tick_data:
                # Tick data resampling logic
                agg_dict = {
                    'price': ['first', 'max', 'min', 'last']
                }
                if 'volume' in self.raw_data.columns:
                    agg_dict['volume'] = 'sum'
                if 'open_interest' in self.raw_data.columns:
                    agg_dict['open_interest'] = 'last'

                temp_resampled = self.raw_data.resample(tf, closed='left', label='left').agg(agg_dict).dropna()

                # Map multi-index columns to standard OHLCV format
                column_mapping = {
                    ('price', 'first'): 'open',
                    ('price', 'max'): 'high',
                    ('price', 'min'): 'low',
                    ('price', 'last'): 'close'
                }
                if 'volume' in self.raw_data.columns:
                    column_mapping[('volume', 'sum')] = 'volume'
                if 'open_interest' in self.raw_data.columns:
                    column_mapping[('open_interest', 'last')] = 'open_interest'

                temp_resampled.columns = pd.MultiIndex.from_tuples(temp_resampled.columns).map(column_mapping)
                resampled[tf] = temp_resampled
            else:
                # Resampling logic for regular data
                agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
                if 'volume' in self.raw_data.columns:
                    agg_dict['volume'] = 'sum'
                if 'open_interest' in self.raw_data.columns:
                    agg_dict['open_interest'] = 'last'

                resampled[tf] = self.raw_data.resample(tf, closed='left').agg(agg_dict).dropna()

        return resampled

    def _infer_frequency(self, datetime_index):
        """
        Infer the frequency of the given datetime index.

        Args:
            datetime_index: Datetime index to infer frequency from.

        Returns:
            Inferred frequency as a string.
        """
        if self.is_tick_data:
            return "TICK"  # No fixed frequency for tick data
        # Try to infer using pandas first
        freq = pd.infer_freq(datetime_index)
        if freq:
            return freq
        # If the data is too incomplete, we use pd.infer_freq on a small chunk
        diffs = datetime_index[1:] - datetime_index[:-1]
        min_delta = diffs.min()
        mask = (diffs == min_delta)[:-1] & (diffs[:-1] == diffs[1:])
        pos = np.where(mask)[0]
        if len(pos) > 0:  # If we found at least one match
            inferred_freq = pd.infer_freq(datetime_index[pos[0]: pos[0] + 3])
            if inferred_freq:
                return inferred_freq
        raise ValueError("Couldn't infer frequency from data! If you have provided tick data, ensure that "
                         "is_tick_data=True upon instantiation.")

    def __iter__(self):
        """
        Initialize the iterator.

        Returns:
            Self.
        """
        return self

    def __next__(self):
        """
        Move to the next data point in the series by updating current_date and current_row.

        Raises:
            StopIteration if the end of the data is reached.
        """
        if self.current_index < len(self.raw_data) - 1:
            self.current_index += 1
            self.current_date = self.raw_data.index[self.current_index]
            self.current_row = {
                timeframe: {
                    column: self.get(column, timeframe)
                    for column in self.data[timeframe].columns
                }
                for timeframe in self.timeframes
            }
        else:
            raise StopIteration

    def _closest_valid_index(self, current_date, timeframe):
        """
        Return the closest valid index for a higher timeframe.

        Args:
            current_date: The current date in consideration.
            timeframe: The higher timeframe.

        Returns:
            Closest valid datetime index for the given timeframe.
        """
        tf_dates = self.data[timeframe].index
        closest_index_position = tf_dates.get_indexer([current_date], method='nearest')[0]
        closest_index_date = tf_dates[closest_index_position]

        if closest_index_date <= current_date:
            # Condition to check if both differences are equal to base_delta
            if (current_date - closest_index_date) == self.base_delta:
                if closest_index_position < len(tf_dates) - 1:
                    next_index_date = tf_dates[closest_index_position + 1]
                    if (next_index_date - current_date) == self.base_delta:
                        return_index_position = closest_index_position
            return_index_position = closest_index_position - 1
        else:
            # Logic for when closest_index_date is more than the current_date
            if (closest_index_date - current_date) == self.base_delta:
                return_index_position = closest_index_position - 1
            else:
                return_index_position = closest_index_position - 2
        if return_index_position < 0:
            return None
        else:
            return tf_dates[return_index_position]

    def get(self, column, timeframe, n=0, future=False):
        """
        Get data from the specified column and timeframe.

        Args:
            column: Column name.
            timeframe: The timeframe.
            n: Number of periods to retrieve.
            future: If data from future is required.

        Returns:
            Value or list of values from the specified column and timeframe.
        """
        current_date = self.data[self.base_timeframe].index[self.current_index]
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe {timeframe} not in loaded data!")

        if column not in self.data[timeframe].columns:
            raise ValueError(f"Column {column} not in loaded data for timeframe {timeframe}!")

        # If n=0, return the current value
        if n == 0:
            if timeframe == self.base_timeframe:
                return self.data[timeframe].iloc[self.current_index][column]
            else:
                closest_valid_date = self._closest_valid_index(current_date, timeframe)
                if closest_valid_date is None:
                    return None
                return self.data[timeframe].loc[closest_valid_date][column]

        # If it's the base timeframe
        if timeframe == self.base_timeframe:
            if future:
                target_idx = self.current_index
                end_idx = target_idx + n
            else:
                target_idx = self.current_index - n
                end_idx = self.current_index + 1
            # Ensure bounds
            target_idx = max(0, target_idx)
            end_idx = min(len(self.data[timeframe]), end_idx)

            return list(self.data[timeframe].iloc[target_idx:end_idx][column])

        # If it's a higher timeframe
        closest_valid_date = self._closest_valid_index(current_date, timeframe)
        if closest_valid_date is None:
            return None

        if future:
            valid_dates = self.data[timeframe].index[
                self.data[timeframe].index >= closest_valid_date]
        else:
            valid_dates = self.data[timeframe].index[
                self.data[timeframe].index <= closest_valid_date]

        end_idx = valid_dates.get_loc(valid_dates[-1])
        start_idx = end_idx - n + 1 if not future else end_idx
        end_idx = start_idx + n

        # Ensure bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data[timeframe]), end_idx)

        return list(self.data[timeframe].iloc[start_idx:end_idx][column])