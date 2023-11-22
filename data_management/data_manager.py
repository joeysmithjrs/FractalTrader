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
from dataclasses import dataclass


def _load_data(data_source, table_name=None) -> pd.DataFrame:
    """
    Load data from various sources into a DataFrame.

    Args:
        data_source (str/pd.DataFrame/np.ndarray/list): Data source to load from. It could be
            a file path (CSV, Excel, SQLite), a DataFrame, a numpy array, or a list of lists.
        table_name (str, optional): Name of the table to query from. Required if data_source is a SQLite database.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.

    Raises:
        ValueError: If the provided data source format is unsupported or issues arise during data loading.
    """
    # Check if data_source is a string (path)
    if isinstance(data_source, str):
        # Load from CSV file
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source, parse_dates=True, index_col=0)
        # Load from Excel file
        elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
            df = pd.read_excel(data_source, parse_dates=True, index_col=0)
        # Load from SQLite database
        elif data_source.endswith('.db'):
            conn = sqlite3.connect(data_source)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn, parse_dates=True, index_col=0)
            conn.close()
        else:
            raise ValueError("Unsupported file format!")
    # Check if data_source is a DataFrame
    elif isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
        # Convert first column to datetime index, if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
                df.drop(df.columns[0], axis=1, inplace=True)
            except:
                raise ValueError("Error converting the first column to datetime!")
    # Check if data_source is a numpy array
    elif isinstance(data_source, np.ndarray):
        df = pd.DataFrame(data_source)
        # Convert first column to datetime index
        try:
            df.set_index(pd.to_datetime(df.iloc[:, 0]), inplace=True)
            df.drop(df.columns[0], axis=1, inplace=True)
        except:
            raise ValueError("Error converting the first column to datetime!")
    # Check if data_source is a list of lists
    elif isinstance(data_source, list):
        if all(isinstance(i, list) for i in data_source):
            df = pd.DataFrame(data_source)
            # Convert first column to datetime index
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
    df.dropna(inplace=True)
    return df


def _ohclv_tick_cols(df : pd.DataFrame) -> pd.DataFrame:
    """
    Rename the columns of a DataFrame based on the type of data (OHLCV or TICK).

    Args:
        df (pd.DataFrame): DataFrame with columns to rename.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    # Define the default column names for OHLCV and TICK data types
    ohlcv_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    tick_cols = ["price", "volume", "open_interest"]

    # Rename based on the number of columns present
    if df.shape[1] <= 3:  # If 3 or fewer columns, assume it's tick data
        df.columns = tick_cols[:df.shape[1]]
    else:  # Assume OHLCV data
        df.columns = ohlcv_cols[:df.shape[1]]
    return df


def _infer_frequency(datetime_index : pd.DatetimeIndex) -> str:
    """
    Deduce the frequency of a datetime index.

    Args:
        datetime_index (pd.DatetimeIndex): Index to infer frequency from.

    Returns:
        str: Inferred frequency.

    Raises:
        ValueError: If the frequency cannot be inferred.
    """
    # Use pandas' built-in function to infer frequency first
    freq = pd.infer_freq(datetime_index)
    if freq:
        return freq

    # If unsuccessful, infer frequency from consistent time deltas in the data
    diffs = datetime_index[1:] - datetime_index[:-1]
    min_delta = diffs.min()
    mask = (diffs == min_delta)[:-1] & (diffs[:-1] == diffs[1:])
    pos = np.where(mask)[0]
    if len(pos) > 0:  # At least one match found
        freq = pd.infer_freq(datetime_index[pos[0]: pos[0] + 3])
        if freq:
            return freq

    raise ValueError("Couldn't infer frequency from data! If you have provided tick data, ensure that "
                     "is_tick_data=True upon instantiation.")


def _freq_to_timedelta(freq : str) -> pd.Timedelta | pd.DateOffset:
    """
    Convert a frequency string to a corresponding timedelta or offset.

    Args:
        freq (str): Frequency string like '5T', 'D', 'M'.

    Returns:
        pd.Timedelta or pd.DateOffset: Corresponding timedelta or offset object.

    Raises:
        ValueError: If the frequency can't be converted to a timedelta.
    """
    # Tick data default
    if freq == "TICK":
        return pd.to_timedelta("1us")

    # For certain frequencies without an initial number, prepend '1' as default
    if not freq[0].isdigit() and freq not in ["M", "Y", "Q", "B"]:
        freq = '1' + freq

    # Convert to timedelta using pandas
    try:
        return pd.to_timedelta(freq)
    except ValueError:
        # Special handling for non-fixed duration frequencies
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


@dataclass
class OHCLV:
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: int = None
    openinterest: int = None

@dataclass
class TICK:
    timeframe: str = "TICK"
    price: float
    volume: int = None
    openinterest: int = None

@dataclass
class MultiRow:
    dt: str
    base: OHCLV | TICK = None 
    timeframe_count: int = 0 

    def __post_init__(self):
        for i in range(1, 101):
            self.__setattr__(self, f't{i}', None)

    def __init__(self, dt, **kwargs):
        self.dt = dt
        super().__init__()
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __setattr__(self, key, value):
        if key.startswith('t') and key[1:].isdigit() and int(key[1:]) <= 100:
            if value is not None and getattr(self, key, None) is None:
                self.timeframe_count += 1
            elif value is None and getattr(self, key, None) is not None:
                self.timeframe_count -= 1
        object.__setattr__(self, key, value)


class MultiFrame:
    """
    A class that represents multiple timeframes of a given dataset.

    Attributes:
        data: Dictionary holding data for different timeframes.
        base_delta: The smallest time difference between data points.
        base_timeframe: The smallest timeframe from which data is derived.
        timeframes: List of all timeframes included in the data.
        current_row: Current row of data.
        current_date: Current date in consideration.
        current_index: Index pointer for iteration purposes.
    """

    def __init__(self):
        """
        Initialize the MultiFrame with default values.
        """
        self.data = {}
        self.base_delta = None
        self.base_timeframe = None
        self.timeframes = None
        self.now = MultiRow()
        self.current_date = None
        self.current_index = -1
        self.tf_date_bounds = {tf : (0, 3) for tf in self.timeframes}
        self.closest_valid_dates = {tf : (None, -1) for tf in self.timeframes}
    
    def __iter__(self):
        """
        Initialize the iterator.

        Returns:
            Self.
        """
        return self
    
    def _sort_timeframes(self, timeframes : list) -> list:
        timeframes_td = [_freq_to_timedelta(tf) for tf in timeframes or []]
        timeframes = [(self.base_timeframe, self.base_delta)] + [(tf, tf_td) for tf, tf_td in zip(timeframes or [], timeframes_td) if
                                                                tf_td >= self.base_delta and tf != self.base_timeframe]
        sorted_timeframes = sorted(timeframes, key=lambda x: x[1])
        return [tf[0] for tf in sorted_timeframes]
 
    def _update_valid_dates(self):
        temp = {tf : (self._closest_valid_index(tf), 0) for tf in self.timeframes}
        for timeframe in self.timeframes[1:]:
            if temp[timeframe][0] != self.closest_valid_dates[timeframe][0]:
                self.closest_valid_dates[timeframe] = temp[timeframe]
            else:
                self.closest_valid_dates[timeframe][1] += 1
                

    def _closest_valid_index(self, timeframe : str) -> pd.DatetimeIndex:
        """
        Return the closest valid index for a higher timeframe based on the current date.

        Args:
            timeframe: The higher timeframe to consider.

        Returns:
            Closest valid datetime index for the given timeframe.
        """
        tf_dates = self.data[timeframe].index[self.tf_date_bounds[timeframe][0], self.tf_date_bounds[timeframe][1]]
        # Find the closest date index in the timeframe's dates to the current date
        closest_index_position = tf_dates.get_indexer([self.current_date], method='nearest')[0]
        closest_index_date = tf_dates[closest_index_position]

        # Check the proximity of the closest date to the current date
        if closest_index_date <= self.current_date:
            if (self.current_date - closest_index_date) == self.base_delta:
                if closest_index_position < len(tf_dates) - 1:
                    next_index_date = tf_dates[closest_index_position + 1]
                    if (next_index_date - self.current_date) == self.base_delta:
                        return_index_position = closest_index_position
            else:
                return_index_position = closest_index_position - 1
        else:
            # Logic for when closest_index_date is greater than the current_date
            if (closest_index_date - self.current_date) == self.base_delta:
                return_index_position = closest_index_position - 1
            else:
                return_index_position = closest_index_position - 2

        self.tf_date_bounds[timeframe][0] = max(return_index_position, 0)
        self.tf_date_bounds[timeframe][1] = min(return_index_position + 1, len(self.data[timeframe].index)-1)
        # Return the found index or None if it's out of bounds
        if return_index_position < 0:
            return None
        else:
            return tf_dates[return_index_position]

    def get(self, column : str, timeframe : str, n : int = 0, future : bool = False) -> float | list:
        """
        Get data from the specified column and timeframe.

        Args:
            column: Column name.
            timeframe: The timeframe to retrieve data from.
            n: Number of periods to retrieve.
            future: If future data is to be considered.

        Returns:
            Value or list of values from the specified column and timeframe.
        """
        # Validate inputs
        if timeframe not in self.timeframes:
            raise ValueError(f"Timeframe {timeframe} not in loaded data!")
        if column not in self.data[timeframe].columns:
            raise ValueError(f"Column {column} not in loaded data for timeframe {timeframe}!")

        # If no periods are specified, return the current value
        if n == 0:
            if timeframe == self.base_timeframe:
                return self.data[timeframe].iloc[self.current_index][column]
            else:
                if self.closest_valid_dates[timeframe][0] is None:
                    return None
                return self.data[timeframe].loc[self.closest_valid_dates[timeframe][0]][column]

        # Handle retrieval from the base timeframe
        if timeframe == self.base_timeframe:
            if future:
                target_idx = self.current_index
                end_idx = target_idx + n
            else:
                target_idx = self.current_index - n
                end_idx = self.current_index + 1

            # Ensure index boundaries are not exceeded
            target_idx = max(0, target_idx)
            end_idx = min(len(self.data[timeframe]), end_idx)

            return list(self.data[timeframe].iloc[target_idx:end_idx][column])

        # Handle retrieval from higher timeframes
        if self.closest_valid_dates[timeframe][0] is None:
            return None
        if future:
            valid_dates = self.data[timeframe].index[self.data[timeframe].index >= self.closest_valid_dates[timeframe][0]]
        else:
            valid_dates = self.data[timeframe].index[self.data[timeframe].index <= self.closest_valid_dates[timeframe][0]]

        # Find the start and end indices to retrieve data
        end_idx = valid_dates.get_loc(valid_dates[-1])
        start_idx = end_idx - n + 1 if not future else end_idx
        end_idx = start_idx + n

        # Ensure index boundaries are not exceeded
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data[timeframe]), end_idx)

        return list(self.data[timeframe].iloc[start_idx:end_idx][column])
    

class PricingSeries(MultiFrame):
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

    def __init__(self, data_source, symbol : str = None, timeframes : list = None, is_tick_data : bool = False, table_name : str = None):
        """
        Initialize the PricingSeries object with raw data, symbol, timeframes, etc.

        Args:
            data_source: Data source can be path to file, DataFrame, numpy array or list.
            symbol: Symbol of the asset.
            timeframes: List of timeframes.
            is_tick_data: If the provided data is tick data.
            table_name: Name of the table in case data_source is a database.
        """
        super().__init__()  # Call the initialization method of the parent class (MultiFrame)

        # Load data and infer important characteristics
        self.raw_data = _ohclv_tick_cols(_load_data(data_source, table_name))
        self.symbol = symbol
        self.is_tick_data = is_tick_data
        self.base_timeframe = _infer_frequency(self.raw_data.index) if not self.is_tick_data else "TICK"
        self.base_delta = _freq_to_timedelta(self.base_timeframe)
        self.timeframes = self._sort_timeframes(timeframes)
        self.bars_since_new = {tf: 0 for tf in self.timeframes}

        # Perform initial data resampling
        self.data = self._resample_data()

    def _resample_data(self) -> dict:
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
                # For tick data, resampling requires different logic

                agg_dict = {
                    'price': ['first', 'max', 'min', 'last']
                }
                if 'volume' in self.raw_data.columns:
                    agg_dict['volume'] = 'sum'
                if 'open_interest' in self.raw_data.columns:
                    agg_dict['open_interest'] = 'last'

                # Resample tick data
                temp_resampled = self.raw_data.resample(tf, closed='left', label='left').agg(agg_dict).dropna()

                # Rename columns to OHLCV format
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
                # For standard data, resampling follows a standard logic
                agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
                if 'volume' in self.raw_data.columns:
                    agg_dict['volume'] = 'sum'
                if 'open_interest' in self.raw_data.columns:
                    agg_dict['open_interest'] = 'last'

                resampled[tf] = self.raw_data.resample(tf, closed='left').agg(agg_dict).dropna()

        return resampled

    def new_bar_open(self, timeframe : str = None) -> bool:
        """
        Check if a new bar has opened for a given timeframe.

        Args:
            timeframe: The timeframe in question.

        Returns:
            Boolean indicating if a new bar has opened.
        """
        if timeframe is None:
            return self.closest_valid_dates[self.base_timeframe][1] == 0
        return self.closest_valid_dates[timeframe][1] == 0

    def __next__(self):
        if self.current_index < len(self.data[self.base_timeframe].index) - 1:
            self.current_index += 1
            self.current_date = self.data[self.base_timeframe].index[self.current_index]
            self._update_valid_dates()

            current_multi_row = MultiRow(dt=self.current_date)

            for idx, timeframe in enumerate(self.timeframes):

                if timeframe == "TICK":
                    timeframe_instance = TICK(price=self.get('price', timeframe=timeframe), 
                                              volume=self.get('volume', timeframe=timeframe) if 'volume' in self.data[timeframe].columns else None,
                                              openinterest=self.get('openinterest', timeframe=timeframe) if 'openinterest' in self.data[timeframe].columns else None)
                else:
                    timeframe_instance = OHCLV(timeframe=timeframe,
                                               open=self.get('open', timeframe),
                                               high=self.get('high', timeframe),
                                               low=self.get('low', timeframe),
                                               close=self.get('close', timeframe),
                                               volume=self.get('volume', timeframe) if 'volume' in self.data[timeframe].columns else None,
                                               openinterest=self.get('openinterest', timeframe) if 'openinterest' in self.data[timeframe].columns else None)
                if idx == 0:
                    current_multi_row.base = timeframe_instance
                else:
                    setattr(current_multi_row, f't{idx}', timeframe_instance)

            self.now = current_multi_row
        else:
            raise StopIteration


class AlternativeSeries(MultiFrame):
    def __init__(self, data_source, pricing_series, agg_dict=None, res_tf=None):
        super().__init__()
        self.raw_data = _load_data(data_source)
        self.alt_timeframe = _infer_frequency(self.raw_data.index)
        self.alt_delta = _freq_to_timedelta(self.alt_timeframe)
        self.agg_dict = agg_dict if agg_dict else {col: 'last' for col in self.raw_data.columns}
        self.res_tf = self.base_timeframe if not res_tf else res_tf
        self.timeframes = pricing_series.timeframes
        self.base_delta = pricing_series.base_delta
        self.base_timeframe = pricing_series.base_timeframe
        self.base_index = pricing_series.data[self.base_timeframe].index
        self.data = self._resample_data()

    def _resample_data(self) -> pd.DataFrame:
        resampled = self.raw_data.resample(self.res_tf, closed='left', label='left').agg(self.agg_dict).dropna()
        return resampled.reindex(self.pricing_series_index, method='ffill')
    
    def is_new_entry(self, timeframe : str = None) -> bool:
        if timeframe is None:
            return self.tf_date_change[self.base_timeframe][1] == 0
        return self.tf_date_change[timeframe][1] == 0
        
    def __next__(self):
        pass

class DataFeed():

    def __init__(self, live=False):
        self.live = live
        self.multiframes = {}
        self.signals = {}

    def __iter__(self):
        return self

    def add_series(self, data : MultiFrame, name : str):
        if isinstance(data, MultiFrame):
            self.multiframes[name] = data
        else:
            raise TypeError(f"Expected PricingSeries or AlternativeSeries, got {type(data).__name__}")

    def add_signal(self, signal, configs):
        self.signals[signal] = [(i.pricing_series, i.timeframe) for i in configs]

    def __next__(self):
        for _, val in self.multiframes:
            val.next()

    def get(self, name, timeframe, n):
        return 

        
