from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(root_path: Path) -> pd.DataFrame:
    # загружаем Копия ЦАТ_общий_parsed_extDS_v6.xlsx в dataframe pandas
    df = pd.read_excel(root_path / 'data/raw/Копия ЦАТ_общий_parsed_extDS_v6.xlsx', decimal=",")

    # загружаем информацию о датах рождения
    birthdates_df = pd.read_excel(root_path / 'data/interim/dataset_wide_birthday.xlsx', decimal=',')

    # проверяем корректность: если id совпал, то и birthdate - тоже
    is_birthdates_equal = birthdates_df.groupby('id')['birthdate'].transform(lambda x: x.nunique() == 1)
    assert is_birthdates_equal.all(), (
        'В данных о датах рождения в строках с одинаковым id'
        ' ожидаются одинаковые даты рождения.'
    )

    # удаляем строки с повторяющимися id
    # (выше проверили, что если id совпал, то и birthdate - тоже)
    unique_birthdates_df = birthdates_df.sort_values('id').drop_duplicates(subset='id')

    # объединяем датафреймы по ключу id, оставляя только колонку birthdate из birthdays_df
    merged_df = df.merge(
        unique_birthdates_df[['id', 'birthdate']], on='id', how='left'
    )
    assert merged_df.shape[0] == df.shape[0], (
        'Ожидается, что при заполнении дат рождения'
        ' количество случаев не изменится.'
    )

    return merged_df


# def remove_misleading_columns(df):
#     """
#     Удалить признаки, использовать которые при предсказании нельзя. Например, id пациента.
#     """
#     return df.drop(['id'], axis=1)


def to_X_y(df):
    """
    Разделение набора данных на свободные переменные X и зависимую переменную y.
    """
    X = df.drop('dose', axis=1)
    y = df['dose']
    return X, y


def custom_train_test_split(df, random_state):
    X, y = to_X_y(df)

    # Разделение данных на обучающий и тестовый наборы (не используя функцию simplified_stratified_split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    return X_train, X_test, y_train, y_test


def change_columns_type_in_df(df, from_dtype: str, to_dtype: str):
    """
    Заменяет типы данных в колонках датафрейма.
    Если не указаны колонки, то меняет типы во всех колонках.
    """
    res_df = df.copy()
    
    columns = res_df.select_dtypes(include=from_dtype).columns      
    res_df[columns] = res_df[columns].astype(to_dtype)

    return res_df


def bool_to_int8_in_df(df):
    """
    Замена False/True на 0/1, пригодтся перез записью в файл.
    """
    return change_columns_type_in_df(df, from_dtype='bool', to_dtype='int8')