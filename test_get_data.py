# test_get_data.py

import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import numpy as np
import pytest
import os
from datetime import datetime, timezone as pytimezone

# Import the script to be tested
import get_data

# --- Fixtures for Test Data ---

@pytest.fixture
def sample_json_data():
    """Provides a sample JSON structure mimicking the Open-Meteo API response."""
    return {
        "latitude": 55.0,
        "longitude": 82.9,
        "elevation": 150.0,
        "timezone": "Asia/Novosibirsk",
        "timezone_abbreviation": "NOVT",
        "utc_offset_seconds": 25200,
        "daily": {
            "time": [1715806800, 1715893200],  # May 16, 2024 & May 17, 2024
            "sunrise": [1715814540, 1715901000], # ~5am
            "sunset": [1715870100, 1715956560], # ~9pm
            "daylight_duration": [55560.0, 55560.0]
        },
        "hourly": {
            "time": [1715835600, 1715839200, 1715922000, 1715925600], # Two hours for each day
            "temperature_2m": [50.0, 53.6, 59.0, 62.6], # Fahrenheit
            "relative_humidity_2m": [80, 75, 70, 65],
            "dew_point_2m": [44.6, 46.4, 50.0, 53.6],
            "rain": [0.0, 0.1, 0.2, 0.0], # Inches
            "showers": [0.0, 0.0, 0.0, 0.0],
            "snowfall": [0.0, 0.0, 0.0, 0.0],
            "wind_speed_10m": [5.0, 6.0, 7.0, 8.0], # Knots
            "visibility": [32808.4, 32808.4, 32808.4, 32808.4] # Feet
        }
    }

@pytest.fixture
def sample_hourly_df():
    """Provides a processed DataFrame for testing aggregate functions."""
    tz_info = pytimezone.utc
    index = pd.to_datetime([
        '2024-05-16 10:00:00', '2024-05-16 23:00:00', # Day 1, one daylight, one night
        '2024-05-17 11:00:00', '2024-05-17 12:00:00'  # Day 2, both daylight
    ]).tz_localize(tz_info)

    sunrise = pd.to_datetime(['2024-05-16 05:00:00', '2024-05-17 05:00:00']).tz_localize(tz_info)
    sunset = pd.to_datetime(['2024-05-16 21:00:00', '2024-05-17 21:00:00']).tz_localize(tz_info)
    
    data = {
        'temperature_2m': [10, 15, 20, 22],
        'relative_humidity_2m': [80, 70, 60, 50],
        'rain': [1.0, 0.0, 2.0, 3.0],
        'snowfall': [0, 0, 0, 0],
        'sunrise': [sunrise[0], sunrise[0], sunrise[1], sunrise[1]],
        'sunset': [sunset[0], sunset[0], sunset[1], sunset[1]]
    }
    df = pd.DataFrame(data, index=index)
    return df


# --- Unit Tests for Individual Functions ---

class TestIOFunctions:
    @patch('get_data.requests.get')
    def test_download_file_success(self, mock_get):
        """Test successful file download."""
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'{"key": "value"}']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        m = mock_open()
        with patch('builtins.open', m):
            get_data.download_file("http://fakeurl.com", {}, "fake/path.json")
            m.assert_called_once_with("fake/path.json", 'wb')
            handle = m()
            handle.write.assert_called_once_with(b'{"key": "value"}')

    @patch('get_data.download_file')
    def test_request_api_json(self, mock_download):
        """Test that API request function calls downloader with correct params."""
        get_data.request_api_json("2024-01-01", "2024-01-02", "path.json", 55.0, 82.9)
        
        args, kwargs = mock_download.call_args
        assert args[0] == get_data.BASE_URL
        assert args[2] == "path.json"
        
        params = args[1]
        assert params['start_date'] == "2024-01-01"
        assert params['end_date'] == "2024-01-02"
        assert params['latitude'] == 55.0
        assert params['longitude'] == 82.9
        assert "temperature_2m" in params['hourly']

    @patch('get_data.pd.DataFrame.to_csv')
    def test_save_to_csv(self, mock_to_csv):
        """Test saving a DataFrame to CSV."""
        df = pd.DataFrame({'a': [1]})
        get_data.save_to_csv(df, 'test.csv')
        mock_to_csv.assert_called_once_with('test.csv', index=False, encoding='utf-8')

    @patch('get_data.execute_values')
    @patch('get_data.psycopg2')
    def test_save_to_db(self, mock_psycopg2, mock_execute_values):
        """Test saving a DataFrame to the database."""
        # Настраиваем поддельный psycopg2
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_psycopg2.connect.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # Создаем тестовый DataFrame
        df = pd.DataFrame({
            'time': ['2024-05-16T17:00:00+07:00'],
            'temperature_2m_celsius': [10.0]
        })

        # Вызываем нашу тестируемую функцию
        get_data.save_to_db(df)

        # --- ПРОВЕРКИ ---

        # 1. Проверяем, что connect был вызван
        mock_psycopg2.connect.assert_called_once()

        # 2. Проверяем, что execute_values (наша вторая "замоканная" функция) была вызвана
        mock_execute_values.assert_called_once()

        # 3. Проверяем аргументы, переданные в execute_values
        call_args = mock_execute_values.call_args[0]
        
        # Первый аргумент - курсор
        assert call_args[0] is mock_cur
        
        # Второй аргумент - SQL-запрос
        sql_arg = call_args[1]
        assert 'INSERT INTO weather_data' in sql_arg
        assert 'ON CONFLICT (time) DO NOTHING' in sql_arg

        # Третий аргумент - данные
        data_arg = call_args[2]
        assert data_arg[0] == ('2024-05-16T17:00:00+07:00', 10.0)

        # 4. Проверяем, что транзакция была закоммичена
        mock_conn.commit.assert_called_once()


class TestTransformationFunctions:

    def test_create_raw_enriched_df(self, sample_json_data):
        """Test initial JSON to DataFrame conversion and enrichment."""
        df = get_data.create_raw_enriched_df(sample_json_data)
        
        assert isinstance(df.index, pd.Index)
        assert df.index.name == 'time'
        assert len(df) == 4
        # Check if daily and metadata are merged correctly
        assert 'daylight_duration' in df.columns
        assert 'latitude' in df.columns
        assert df['latitude'].iloc[0] == 55.0
        # Check merge_asof logic: first hourly data point should get first daily data
        assert df['daylight_duration'].iloc[0] == 55560.0

    def test_convert_units(self):
        """Test conversion from imperial to metric units."""
        df = pd.DataFrame({
            'temperature_2m': [32.0],        # F -> C (0)
            'wind_speed_10m': [1.94384],      # knots -> m/s (1)
            'rain': [0.03937],               # inches -> mm (1)
            'visibility': [3280.84]          # feet -> meters (1000)
        })
        
        df_converted = get_data.convert_units(df)
        
        assert np.isclose(df_converted['temperature_2m'].iloc[0], 0.0)
        assert np.isclose(df_converted['wind_speed_10m'].iloc[0], 1.0)
        assert np.isclose(df_converted['rain'].iloc[0], 1.0)
        assert np.isclose(df_converted['visibility'].iloc[0], 1000.0)

    def test_process_datetime_columns(self, sample_json_data):
        """Test conversion of unix timestamps to datetime objects."""
        raw_df = get_data.create_raw_enriched_df(sample_json_data)
        df_processed = get_data.process_datetime_columns(raw_df)

        assert isinstance(df_processed.index, pd.DatetimeIndex)
        assert str(df_processed.index.tz) == 'Asia/Novosibirsk'
        assert isinstance(df_processed['sunrise'].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype)
        assert 'daylight_hours' in df_processed.columns
        assert np.isclose(df_processed['daylight_hours'].iloc[0], 55560.0 / 3600)
        assert 'daylight_duration' not in df_processed.columns

    def test_add_24h_aggregates(self, sample_hourly_df):
        """Test calculation of 24-hour aggregates."""
        # Manually set AGG_RULES for this test to match sample_hourly_df
        with patch.dict(get_data.AGG_RULES, {
            'temperature_2m': 'mean',
            'relative_humidity_2m': 'mean',
            'rain': 'sum'
        }, clear=True):
            df_agg = get_data.add_24h_aggregates(sample_hourly_df)

        # Expected values for day 1 (2024-05-16)
        expected_temp_d1 = (10 + 15) / 2
        expected_hum_d1 = (80 + 70) / 2
        expected_rain_d1 = 1.0 + 0.0
        
        # Expected values for day 2 (2024-05-17)
        expected_temp_d2 = (20 + 22) / 2
        expected_hum_d2 = (60 + 50) / 2
        expected_rain_d2 = 2.0 + 3.0
        
        assert 'avg_temperature_2m_24h' in df_agg.columns
        assert np.isclose(df_agg['avg_temperature_2m_24h'].iloc[0], expected_temp_d1)
        assert np.isclose(df_agg['total_rain_24h'].iloc[0], expected_rain_d1)
        assert np.isclose(df_agg['avg_relative_humidity_2m_24h'].iloc[0], expected_hum_d1)
        
        assert np.isclose(df_agg['avg_temperature_2m_24h'].iloc[2], expected_temp_d2)
        assert np.isclose(df_agg['total_rain_24h'].iloc[2], expected_rain_d2)
        assert np.isclose(df_agg['avg_relative_humidity_2m_24h'].iloc[2], expected_hum_d2)

    def test_add_daylight_aggregates(self, sample_hourly_df):
        """Test calculation of daylight-only aggregates."""
        with patch.dict(get_data.AGG_RULES, {
            'temperature_2m': 'mean',
            'rain': 'sum'
        }, clear=True):
            df_agg = get_data.add_daylight_aggregates(sample_hourly_df)
        
        # Day 1: only the 10:00 record is daylight.
        expected_temp_d1 = 10.0
        expected_rain_d1 = 1.0
        
        # Day 2: both records (11:00, 12:00) are daylight.
        expected_temp_d2 = (20 + 22) / 2
        expected_rain_d2 = 2.0 + 3.0
        
        assert 'avg_temperature_2m_daylight' in df_agg.columns
        assert np.isclose(df_agg['avg_temperature_2m_daylight'].iloc[0], expected_temp_d1)
        assert np.isclose(df_agg['total_rain_daylight'].iloc[0], expected_rain_d1)
        
        assert np.isclose(df_agg['avg_temperature_2m_daylight'].iloc[2], expected_temp_d2)
        assert np.isclose(df_agg['total_rain_daylight'].iloc[2], expected_rain_d2)

    def test_rename_final_columns(self):
        """Test renaming of columns for the final output."""
        df = pd.DataFrame(columns=['temperature_2m', 'wind_speed_10m', 'sunrise'])
        df_renamed = get_data.rename_final_columns(df)
        
        assert 'temperature_2m_celsius' in df_renamed.columns
        assert 'wind_speed_10m_m_per_s' in df_renamed.columns
        assert 'sunrise_iso' in df_renamed.columns
        
    def test_clean_raw_data(self):
        """Test cleaning of rows with all-NaN key columns."""
        df = pd.DataFrame({
            'temperature_2m': [10, np.nan, 20, np.nan],
            'rain': [0.1, np.nan, 0.5, np.nan],
            'other_col': [1, 2, 3, 4]
        })
        key_cols = ['temperature_2m', 'rain']
        
        df_cleaned = get_data.clean_raw_data(df, key_cols)
        assert len(df_cleaned) == 2
        assert df_cleaned.index.tolist() == [0, 2]

class TestMainPipeline:

    @patch('get_data.save_to_db')
    @patch('get_data.save_to_csv')
    @patch('get_data.transform_json_csv')
    @patch('get_data.request_api_json')
    @patch('get_data.argparse.ArgumentParser.parse_args')
    def test_main_flow_api_to_db(self, mock_args, mock_request, mock_transform, mock_csv, mock_db):
        """Test the main function flow: api -> db."""
        mock_args.return_value = MagicMock(source='api', destination='db', start='s', end='e',
                                           latitude=1.0, longitude=1.0, filepath='f.json')
        mock_transform.return_value = pd.DataFrame({'a': [1]}) # Non-empty DataFrame
        
        get_data.main()
        
        mock_request.assert_called_once()
        mock_transform.assert_called_once()
        mock_db.assert_called_once()
        mock_csv.assert_not_called()
        
    @patch('get_data.save_to_db')
    @patch('get_data.save_to_csv')
    @patch('get_data.transform_json_csv')
    @patch('get_data.request_api_json')
    @patch('get_data.argparse.ArgumentParser.parse_args')
    def test_main_flow_file_to_csv(self, mock_args, mock_request, mock_transform, mock_csv, mock_db):
        """Test the main function flow: file -> csv."""
        mock_args.return_value = MagicMock(source='file', destination='csv', start='s', end='e',
                                           latitude=1.0, longitude=1.0, filepath='f.json')
        mock_transform.return_value = pd.DataFrame({'a': [1]}) # Non-empty DataFrame

        get_data.main()

        mock_request.assert_not_called()
        mock_transform.assert_called_once()
        mock_db.assert_not_called()
        mock_csv.assert_called_once()

    @patch('get_data.save_to_db')
    @patch('get_data.save_to_csv')
    @patch('get_data.transform_json_csv')
    @patch('get_data.request_api_json')
    @patch('get_data.argparse.ArgumentParser.parse_args')
    def test_main_flow_empty_dataframe(self, mock_args, mock_request, mock_transform, mock_csv, mock_db):
        """Test that save functions are not called if the dataframe is empty."""
        mock_args.return_value = MagicMock(source='api', destination='db', start='s', end='e',
                                           latitude=1.0, longitude=1.0, filepath='f.json')
        mock_transform.return_value = pd.DataFrame() # Empty DataFrame

        get_data.main()
        
        mock_request.assert_called_once()
        mock_transform.assert_called_once()
        mock_db.assert_not_called()
        mock_csv.assert_not_called()

    @patch('json.load')
    def test_transform_json_csv_full_pipeline(self, mock_json_load, sample_json_data):
        """An integration test for the main transformation pipeline."""
        mock_json_load.return_value = sample_json_data
        
        # Use a mock open since the file path is irrelevant with mock_json_load
        with patch('builtins.open', mock_open(read_data='{}')):
            final_df = get_data.transform_json_csv("dummy_path.json")

        # Check for final structure and some values
        assert not final_df.empty
        assert 'avg_temperature_2m_24h' in final_df.columns
        assert 'total_rain_daylight' in final_df.columns
        assert 'temperature_2m_celsius' in final_df.columns
        assert 'time' in final_df.columns
        assert final_df['time'].iloc[0] == '2024-05-16T12:00:00+07:00'
        
        # Check a converted value. 50F -> 10C
        assert np.isclose(final_df['temperature_2m_celsius'].iloc[0], 10.0)
        # Check a 24h aggregate. Day 1 temps: 50F, 53.6F -> 10C, 12C. Avg = 11C
        assert np.isclose(final_df['avg_temperature_2m_24h'].iloc[0], 11.0)