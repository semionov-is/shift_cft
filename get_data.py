import pandas as pd
import json
import requests
import psycopg2
from psycopg2.extras import execute_values
import os
import argparse
from datetime import datetime
from typing import Union, Dict, Any
from pandas.api.types import is_numeric_dtype


BASE_URL = "https://api.open-meteo.com/v1/forecast"

KNOTS_TO_MPS = 0.514444
INCHES_TO_MM = 25.4
FEET_TO_METERS = 0.3048

# устанавливаем, до какого количества знаков после запятой будем округлять
R_VAL = 4

TEMPERATURE_COLS = [
    'temperature_2m', 'dew_point_2m', 'apparent_temperature',
    'temperature_80m', 'temperature_120m', 'soil_temperature_0cm',
    'soil_temperature_6cm'
    ]
WIND_SPEED_COLS = ['wind_speed_10m', 'wind_speed_80m']
PRECIPITATION_COLS = ['rain', 'showers', 'snowfall', 'evapotranspiration']
DISTANCE_COLS = ['visibility']

AGG_RULES = {
    'temperature_2m': 'mean', 'relative_humidity_2m': 'mean',
    'dew_point_2m': 'mean', 'apparent_temperature': 'mean',
    'temperature_80m': 'mean', 'temperature_120m': 'mean',
    'wind_speed_10m': 'mean', 'wind_speed_80m': 'mean',
    'visibility': 'mean', 'rain': 'sum', 'showers': 'sum', 'snowfall': 'sum'
    }

FINAL_COLS_ORDER = [
    'avg_temperature_2m_24h', 'avg_relative_humidity_2m_24h', 'avg_dew_point_2m_24h',
    'avg_apparent_temperature_24h', 'avg_temperature_80m_24h', 'avg_temperature_120m_24h',
    'avg_wind_speed_10m_24h', 'avg_wind_speed_80m_24h', 'avg_visibility_24h', 'total_rain_24h',
    'total_showers_24h', 'total_snowfall_24h', 'avg_temperature_2m_daylight',
    'avg_relative_humidity_2m_daylight', 'avg_dew_point_2m_daylight', 'avg_apparent_temperature_daylight',
    'avg_temperature_80m_daylight', 'avg_temperature_120m_daylight', 'avg_wind_speed_10m_daylight',
    'avg_wind_speed_80m_daylight', 'avg_visibility_daylight', 'total_rain_daylight', 'total_showers_daylight',
    'total_snowfall_daylight', 'wind_speed_10m_m_per_s', 'wind_speed_80m_m_per_s', 'temperature_2m_celsius',
    'apparent_temperature_celsius', 'temperature_80m_celsius', 'temperature_120m_celsius',
    'soil_temperature_0cm_celsius', 'soil_temperature_6cm_celsius', 'rain_mm', 'showers_mm',
    'snowfall_mm', 'daylight_hours', 'sunset_iso', 'sunrise_iso'
    ]


def download_file(url: str, params: Dict[str, Any], save_path: str) -> None:
    """
    Скачивает JSON-файл по заданному URL и сохраняет по указанному пути.
    """
    
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Файл JSON успешно скачен и сохранен: {save_path}")
    
    except requests.exceptions.RequestException as e:
        print(f"Ошибка скачивания JSON-файла: {e}")


def request_api_json(start_date: str, end_date: str, save_path: str, latitude: float, longitude: float):
    """
    Создаёт запрос к Open-Meteo API и вызывает функцию для получения данных в json.
    """

    hourly_params = (
        "temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,"
        "temperature_80m,temperature_120m,wind_speed_10m,wind_speed_80m,"
        "wind_direction_10m,wind_direction_80m,visibility,evapotranspiration,"
        "weather_code,soil_temperature_0cm,soil_temperature_6cm,rain,showers,snowfall"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "sunrise,sunset,daylight_duration",
        "hourly": hourly_params,
        "timezone": "auto",
        "timeformat": "unixtime",
        "wind_speed_unit": "kn",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "start_date": start_date,
        "end_date": end_date
    }
    
    download_file(BASE_URL, params, save_path)


def create_raw_enriched_df(json_data: Union[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Преобразует JSON-данные в DataFrame, обогащая почасовые записи суточными и метаданными.
    Все значения остаются в их исходном формате, включая временные метки (unixtime).
    """

    if isinstance(json_data, str):
        with open(json_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = json_data

    hourly_df = pd.DataFrame(data['hourly'])
    daily_df = pd.DataFrame(data['daily'])

    # Переименуем 'time' в daily_df, чтобы избежать конфликта имен после слияния
    daily_df.rename(columns={'time': 'daily_time_key'}, inplace=True)

    hourly_df.sort_values('time', inplace=True)
    daily_df.sort_values('daily_time_key', inplace=True)

    enriched_df = pd.merge_asof(
        left=hourly_df,
        right=daily_df,
        left_on='time',           # Ключ из почасового hourly_df
        right_on='daily_time_key',# Ключ из суточного daily_df
        direction='backward'
    )

    # Добавление общих метаданных
    metadata_keys = [
        'latitude', 'longitude', 'elevation', 'timezone',
        'timezone_abbreviation', 'utc_offset_seconds'
        ]
    for key in metadata_keys:
        if key in data:
            enriched_df[key] = data[key]

    # Удаляем вспомогательный ключ слияния
    enriched_df.drop(columns=['daily_time_key'], inplace=True)
    # Устанавливаем почасовой unixtime (!) как индекс
    enriched_df.set_index('time', inplace=True)

    return enriched_df


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет преобразование всех единиц из имперских в метрические.
    - температура: Фаренгейт -> Цельсий
    - скорость: узлы -> м/с
    - осадки: дюймы -> мм
    - видимость: футы -> метры
    """
    
    df = df.copy()

    for col in TEMPERATURE_COLS:
        if col in df.columns:
            df[col] = ((df[col] - 32) * 5 / 9).round(R_VAL)

    for col in WIND_SPEED_COLS:
        if col in df.columns:
            df[col] = (df[col] * KNOTS_TO_MPS).round(R_VAL)

    for col in PRECIPITATION_COLS:
        if col in df.columns:
            df[col] = (df[col] * INCHES_TO_MM).round(R_VAL)

    for col in DISTANCE_COLS:
        if col in df.columns:
            df[col] = (df[col] * FEET_TO_METERS).round(R_VAL)

    return df


def process_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует временные колонки из unixtime в datetime объекты
    с корректной временной зоной и рассчитывает длительность светового дня.
    """
    
    df = df.copy()

    if 'timezone' not in df.columns or df['timezone'].isnull().all():
        timezone = 'UTC'
    else:
        timezone = df['timezone'].iloc[0]

    # Преобразование времени восхода/заката
    for col in ['sunrise', 'sunset']:
        if col in df.columns:
            if is_numeric_dtype(df[col]):
                # Unix time -> datetime с временной зоной
                naive_dt = pd.to_datetime(df[col], unit='s')
                utc_dt = naive_dt.dt.tz_localize('UTC')
                df[col] = utc_dt.dt.tz_convert(timezone)

    # Преобразование индекса
    if is_numeric_dtype(df.index):
        df.index = pd.to_datetime(df.index, unit='s').tz_localize('UTC').tz_convert(timezone)

    # Считаем длительность светового дня
    if 'daylight_duration' in df.columns:
        df['daylight_hours'] = (df['daylight_duration'] / 3600).round(R_VAL)
        df = df.drop(columns=['daylight_duration'])

    return df


def set_column_types_and_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Упорядочивает столбцы в нужном порядке.
    """
    
    df = df.copy()

    ordered_columns = [
        # Температурные метрики
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'temperature_80m', 'temperature_120m',
        # Расширенные метрики
        'wind_speed_10m', 'wind_direction_10m', 
        'wind_speed_80m', 'wind_direction_80m',
        'visibility', 'evapotranspiration', 'weather_code',
        'soil_temperature_0cm', 'soil_temperature_6cm', 
        # Осадки
        'rain', 'showers', 'snowfall',
        # Суточные и метаданные
        'sunset', 'sunrise', 'daylight_hours',
        'latitude', 'longitude', 'elevation', 'timezone'
    ]
    
    # Убираем колонки, которых нет в df, и добавляем те, что не учли, в конец
    final_cols = [col for col in ordered_columns if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in final_cols]
    
    df = df[final_cols + remaining_cols]

    # Убираем колонки, ненужные на следующем этапе
    remove_cols = ['wind_direction_10m', 'wind_direction_80m', 'evapotranspiration',
                   'weather_code', 'latitude', 'longitude', 'elevation', 'timezone',
                   'timezone_abbreviation', 'utc_offset_seconds']
    for col in remove_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def optimize_dtypes(df: pd.DataFrame, verbose=True) -> pd.DataFrame:
    """
    Оптимизирует типы данных в DataFrame, гарантированно исключая временные колонки и индекс.
    Параметр verbose  (bool, default=True) определяет, выводить ли информацию об оптимизации.
    """
    
    df = df.copy()
    
    original_memory = df.memory_usage(deep=True).sum()
    
    # Сохраняем исходный индекс
    original_index = df.index
    time_is_index = False
    
    # Проверяем, является ли 'time' индексом
    if df.index.name == 'time' or (isinstance(df.index, pd.DatetimeIndex) and df.index.name is None):
        time_is_index = True
        if verbose:
            print("Обнаружено, что 'time' является индексом - не будет оптимизирован")
    
    # Явный список временнЫх колонок для исключения
    time_related_cols = ['time', 'sunrise', 'sunset', 'daylight_hours']
    # Собираем колонки для оптимизации (исключая временнЫе)
    cols_to_optimize = [col for col in df.columns if col not in time_related_cols]
    # Создаем копию DataFrame для безопасной оптимизации
    df_optimized = df.copy()
    
    # Оптимизация float-колонок
    float_cols = df_optimized[cols_to_optimize].select_dtypes(include=['float']).columns
    for col in float_cols:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
    # Оптимизация integer-колонок
    int_cols = df_optimized[cols_to_optimize].select_dtypes(include=['integer']).columns
    for col in int_cols:
        col_min = df_optimized[col].min()
        col_max = df_optimized[col].max()
        
        if col_min >= 0:  # Беззнаковые типы
            if col_max < 256:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='unsigned')
            elif col_max < 65536:
                df_optimized[col] = df_optimized[col].astype('uint16')
        else:  # Знаковые типы
            if col_min > -129 and col_max < 128:
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            elif col_min > -32769 and col_max < 32768:
                df_optimized[col] = df_optimized[col].astype('int16')

    # Восстанавливаем исходный индекс
    df_optimized.index = original_index
    
    if verbose:
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        reduction = (original_memory - optimized_memory) / original_memory * 100
        print(f"Оптимизация завершена. Экономия памяти: {reduction:.2f}%")
        
        excluded = [col for col in time_related_cols if col in df.columns]
        if time_is_index:
            excluded.append('time (index)')
        print("Исключенные колонки:", excluded)
        print("Итоговые типы данных:\n", df_optimized.dtypes)
    
    return df_optimized


def add_24h_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обогащение обработанных данных средними данными за 24 часа.
    """

    df_processed = df.copy()

    if not isinstance(df_processed.index, pd.DatetimeIndex):
        raise TypeError("Индекс DataFrame должен быть типа DatetimeIndex для этой функции.")

    # Оставим только те правила, для которых есть колонки в df И ЕСЛИ в них есть хоть одно не-null значение
    valid_agg_rules = {
        k: v 
        for k, v in AGG_RULES.items() 
        if k in df_processed.columns and df_processed[k].notna().any()
    }

    # Если после фильтрации не осталось колонок для агрегации, просто возвращаем исходный DataFrame
    if not valid_agg_rules:
        print("Предупреждение: Нет данных для 24-часовой агрегации. Пропускаем шаг.")
        return df_processed

    # Индексом в daily_avg будут объекты date
    daily_avg = df_processed.groupby(df_processed.index.date).agg(valid_agg_rules).round(R_VAL)

    new_column_names = {}
    for col, agg_func in valid_agg_rules.items():
        prefix = 'avg' if agg_func == 'mean' else 'total'
        new_column_names[col] = f"{prefix}_{col}_24h"

    daily_avg = daily_avg.rename(columns=new_column_names)

    # Создаем временную колонку с датой в основном DataFrame для ключа слияния.
    df_processed['merge_date'] = df_processed.index.date
    # Объединяем df по колонке 'merge_date', а daily_avg - по его индексу.
    df_with_daily = df_processed.merge(daily_avg, left_on='merge_date', right_index=True, how='left')
    # Удаляем временную колонку.
    df_with_daily = df_with_daily.drop(columns=['merge_date'])
    
    print("DataFrame успешно обогащен средними данными за 24 часа.")
    
    return df_with_daily


def add_daylight_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Обогащает DataFrame агрегированными данными за период светового дня: для каждых суток вычисляются 
    средние значения для температуры, влажности, скорости ветра и видимости, а также суммы осадков,
    используя только те часовые измерения, которые попадают в интервал времени [sunrise, sunset].
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Индекс DataFrame должен быть типа DatetimeIndex.")
    if not all(col in df.columns for col in ['sunrise', 'sunset']):
        raise ValueError("В DataFrame отсутствуют обязательные колонки 'sunrise' и 'sunset'.")
        
    df_copy = df.copy()
    df_copy['sunrise'] = pd.to_datetime(df_copy['sunrise'])
    df_copy['sunset'] = pd.to_datetime(df_copy['sunset'])

    # Создаем булеву "маску", где True для строк внутри светового дня
    daylight_mask = (df_copy.index >= df_copy['sunrise']) & (df_copy.index <= df_copy['sunset'])
    # Применяем маску, чтобы получить DataFrame только с "дневными" данными
    daylight_df = df_copy[daylight_mask]
    
    # Если в дневные часы не попало ни одной записи, возвращаем исходный DataFrame
    if daylight_df.empty:
        print("Предупреждение: Нет данных за световой день для агрегации.")
        # Создаем пустые колонки, чтобы структура DataFrame не менялась
        for col, agg_func in AGG_RULES.items():
             prefix = 'avg' if agg_func == 'mean' else 'total'
             df_copy[f"{prefix}_{col}_daylight"] = pd.NA
        return df_copy
    
    # Оставим только те агрегации, для которых есть колонки в df И ЕСЛИ в них есть хоть одно не-null значение
    valid_aggregations = {
        k: v 
        for k, v in AGG_RULES.items() 
        if k in df_copy.columns and daylight_df[k].notna().any()
    }
    
    if not valid_aggregations:
        print("Внимание: В DataFrame нет колонок для расчета дневных агрегатов.")
        for col, agg_func in AGG_RULES.items():
             prefix = 'avg' if agg_func == 'mean' else 'total'
             df_copy[f"{prefix}_{col}_daylight"] = pd.NA
        return df_copy

    # Группируем "дневные" данные по дате и вычисляем агрегаты
    daily_stats = daylight_df.groupby(daylight_df.index.date).agg(valid_aggregations)
    daily_stats = daily_stats.round(R_VAL)
    
    # Переименование новых колонок
    new_column_names = {}
    for col, agg_func in valid_aggregations.items():
        prefix = 'avg' if agg_func == 'mean' else 'total'
        new_column_names[col] = f"{prefix}_{col}_daylight"
    daily_stats = daily_stats.rename(columns=new_column_names)

    # Присоединяем вычисленные дневные статистики к исходному почасовому DataFrame
    # Ключ для соединения - дата (df.index.date)
    df_enriched = df_copy.join(daily_stats, on=df_copy.index.date)
    
    print("DataFrame успешно обогащен данными за световой день.")
    return df_enriched


def rename_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переименовывает колонки DataFrame в соответствии со списком.
    """

    df = df.copy()
    to_rename = ['wind_speed_10m', 'wind_speed_80m', 'temperature_2m', 'apparent_temperature', 'temperature_80m',
                 'temperature_120m', 'soil_temperature_0cm', 'soil_temperature_6cm', 'rain', 'showers',
                 'snowfall', 'sunset', 'sunrise']
    new_names = ['wind_speed_10m_m_per_s', 'wind_speed_80m_m_per_s', 'temperature_2m_celsius',
                 'apparent_temperature_celsius', 'temperature_80m_celsius', 'temperature_120m_celsius',
                 'soil_temperature_0cm_celsius', 'soil_temperature_6cm_celsius', 'rain_mm', 'showers_mm',
                 'snowfall_mm', 'sunset_iso', 'sunrise_iso']
    rename_dict = dict(zip(to_rename, new_names))

    df = df.rename(columns=rename_dict)
    return df
    
    
def reorder_final_columns(df: pd.DataFrame, final_column_order: list) -> pd.DataFrame:
    """
    Упорядочивает колонки DataFrame в соответствии с заданным списком.
    """

    df = df.copy()
    full_order = [col for col in final_column_order if col in df.columns]
    
    return df[full_order]


def timedate_to_iso8601(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует формат колонок 'time', 'sunrise_iso', 'sunset_iso' для итогового форматирования в ISO 8601.
    """
    df = df.copy()
    for col in ['time', 'sunrise_iso', 'sunset_iso']:
        if col in df.columns:
            # .isoformat() автоматически создаст строку в нужном формате (e.g., '2024-05-21T04:00:00+07:00')
            df[col] = df[col].apply(lambda dt: dt.isoformat())
    return df


def save_to_db(df: pd.DataFrame):
    """
    Saves a DataFrame to the PostgreSQL database.
    """
    
    df = df.copy()
    
    if df.empty:
        print("DataFrame is empty. Nothing to save to the database.")
        return

    # Get DB credentials from environment variables
    DB_HOST = os.getenv("DB_HOST", "postgres")
    DB_NAME = os.getenv("DB_NAME", "weather_db")
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "p@$$w0rd")
    DB_TABLE = 'weather_data'
    
    # Convert DataFrame to a list of tuples for psycopg2
    # The 'time' column should be the first one after reset_index()
    header = df.columns.tolist()
    data_rows = list(df.itertuples(index=False, name=None))
    
    conn = None
    try:
        conn = psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS)
        with conn.cursor() as cur:
            column_names = ', '.join([f'"{h}"' for h in header])
            # Для ON CONFLICT нужно указать поле, которое является уникальным - 'time'
            sql_insert = (
                f"INSERT INTO {DB_TABLE} ({column_names}) VALUES %s "
                f"ON CONFLICT (time) DO NOTHING"
            )
            
            # execute_values выполняет пакетную вставку
            execute_values(cur, sql_insert, data_rows)
            inserted_count = cur.rowcount
            
            conn.commit()
            print(f"Batch insert completed. Inserted {inserted_count} new rows into '{DB_TABLE}'.")

    except Exception as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback() # Откатываем транзакцию в случае ошибки
    finally:
        if conn:
            conn.close()


def save_to_csv(df: pd.DataFrame, file_path: str):
    """
    Сохраняет DataFrame в CSV файл.
    """
    
    df = df.copy()
    
    if df.empty:
        print("DataFrame is empty. Nothing to save to CSV.")
        return
    
    try:
        df.to_csv(file_path, index=False, encoding='utf-8')
        # df.to_csv(file_path, index=True, encoding='utf-8')
        print(f"DataFrame успешно сохранен в файл: '{file_path}'")
    except Exception as e:
        print(f"Ошибка при сохранении файла '{file_path}': {e}")


def clean_raw_data(df: pd.DataFrame, key_columns: list) -> pd.DataFrame:
    """
    Удаляет строки, где все ключевые столбцы имеют значение NaN.
    """
    initial_rows = len(df)
    # Удаляем строки, только если ВСЕ указанные колонки являются NaN
    df_cleaned = df.dropna(subset=key_columns, how='all')
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
        print(f"Очистка данных: Удалено {rows_dropped} строк из-за отсутствия данных в ключевых колонках.")
    return df_cleaned
    

def transform_json_csv(file_path: str) -> pd.DataFrame:
    """
    Проводит нужные преобразования из JSON в CSV файл.
    """
    
    raw_df = create_raw_enriched_df(file_path)
    
    # Определяем ключевые колонки. Если в строке нет ни одного из этих значений, она бесполезна.
    key_weather_columns = [
        # Температурные метрики
        'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
        'apparent_temperature', 'temperature_80m', 'temperature_120m',
        # Осадки
        'rain', 'showers', 'snowfall'
    ]
    
    # Убедимся, что мы проверяем только те колонки, которые есть в DataFrame
    cols_to_check = [col for col in key_weather_columns if col in raw_df.columns]
    cleaned_df = raw_df.pipe(clean_raw_data, key_columns=cols_to_check)

    # Если после очистки DataFrame стал пустым, нет смысла продолжать
    if cleaned_df.empty:
        print("Внимание: После очистки не осталось данных для обработки.")
        return pd.DataFrame() # Возвращаем пустой DataFrame
        
    df = (cleaned_df
        .pipe(convert_units)
        .pipe(process_datetime_columns)
        .pipe(set_column_types_and_order)
        .pipe(optimize_dtypes, verbose=False)    # Оптимизация типов переменных - уменьшение размера df, по желанию
        .pipe(add_24h_aggregates)
        .pipe(add_daylight_aggregates)
        .pipe(rename_final_columns)
        .pipe(reorder_final_columns, FINAL_COLS_ORDER).reset_index()
        .pipe(timedate_to_iso8601)
    )
    
    return df


def main():
    """
    Главная функция, запускающая ETL-пайплайн.
    """
    
    parser = argparse.ArgumentParser(description="Flexible ETL tool for weather data.")
    
    parser.add_argument('source', choices=['api', 'file'], help="Data source: 'api' or 'file'.")
    parser.add_argument('destination', choices=['db', 'csv'], help="Data destination: 'db' or 'csv'.")
    
    parser.add_argument('--start', default='2025-05-16', help="Start date for API fetch (YYYY-MM-DD).")
    parser.add_argument('--end', default='2025-05-30', help="End date for API fetch (YYYY-MM-DD).")
    
    # Координаты места для возможного расширения проекта
    parser.add_argument('--latitude', type=float, default=55.0344, help="Latitude for the API request.")
    parser.add_argument('--longitude', type=float, default=82.9434, help="Longitude for the API request.")
    
    parser.add_argument('--filepath', default='/app/data/weather_data.json', help="Path to source JSON file.")
    
    args = parser.parse_args()

    OUTPUT_DIR = "data"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    dataframe = pd.DataFrame()

    JSON_SAVE_PATH = os.path.join(OUTPUT_DIR, "weather_data.json")
    FINAL_CSV_PATH = os.path.join(OUTPUT_DIR, "final_data.csv")

    try:
        if args.source == 'api':
            print(f"Источник: Метео API. Получаем данные с {args.start} по {args.end}.")
            # Скачиваем и сохраняем файл json
            request_api_json(
                start_date=args.start, 
                end_date=args.end, 
                save_path=JSON_SAVE_PATH,
                # координаты для расширения проекта, если применять - нужно менять схему БД
                latitude=args.latitude, 
                longitude=args.longitude
            )
            
            dataframe = transform_json_csv(JSON_SAVE_PATH)
            
        elif args.source == 'file':
            # Берём данные из файла json
            print(f"Источник: файл JSON. Загружаем из {args.filepath}...")
            try:
                dataframe = transform_json_csv(JSON_SAVE_PATH)
                print(f"Успешно прочитано {len(dataframe)} строк из {args.filepath}")
            except FileNotFoundError:
                print(f"ОШИБКА: Файл не найден в {args.filepath}")

        if not dataframe.empty:
            # Загружаем в БД
            if args.destination == 'db':
                print("Получатель: база данных. Загружаем данные.")
                save_to_db(dataframe)
            # Сохраняем в csv
            elif args.destination == 'csv':
                print("Получатель: файл CSV. Сохраняем данные.")
                save_to_csv(dataframe, FINAL_CSV_PATH)
        else:
            print("Не было данных для загрузки или обработки. Завершение работы.")
            
        print("Процесс ETL завершён.")

    except Exception as e:
        print(f"\nПроизошла непредвиденная ошибка: {e}")


if __name__ == "__main__":
    main()
