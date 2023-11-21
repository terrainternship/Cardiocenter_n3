from pathlib import Path

import pandas as pd



def load_raw_data(root_path: Path) -> pd.DataFrame:
    # загружаем Копия ЦАТ_общий_parsed_extDS_v6.xlsx в dataframe pandas
    df = pd.read_excel(root_path / 'data/raw/Копия ЦАТ_общий_parsed_extDS_v6.xlsx', decimal=",")

    # загружаем информацию о датах рождения, которая ранее создалась отдельным скриптом
    birthdates_df = pd.read_excel(root_path / 'data/interim/dataset_wide_birthday.xlsx', decimal=',')

    # проверяем корректность: если id совпал, то и birthdate - тоже
    is_birthdates_equal = birthdates_df.groupby('id')['birthdate'].transform(lambda x: x.nunique() == 1)
    if not is_birthdates_equal.all():
        raise ValueError(
            'В данных о датах рождения в строках с одинаковым id ожидаются одинаковые даты рождения.'
        )

    # удаляем строки с повторяющимися id
    # (выше проверили, что если id совпал, то и birthdate - тоже)
    unique_birthdates_df = birthdates_df.sort_values('id').drop_duplicates(subset='id')

    # объединяем датафреймы по ключу id, оставляя только колонку birthdate из birthdays_df
    merged_df = df.merge(
        unique_birthdates_df[['id', 'birthdate']], on='id', how='left'
    )
    if merged_df.shape[0] != df.shape[0]:
        raise ValueError(
            'Ожидается, что при заполнении дат рождения количество случаев не изменится.'
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


def _drop_na_rows(df):
    """
    Удаляем строки, в которых есть пропуски.
    Учитываем, что колонка 'sss' может быть не заполнена, это нормально.
    """
    # создаем копию датафрейма без колонки 'sss'
    copy_df = df.drop('sss', axis=1)

    # находим индексы строк, где есть na
    na_rows = copy_df.apply(pd.isna).any(axis=1)

    # удаляем эти строки из исходного датафрейма
    return df[~na_rows]


def _drop_invalid_age_rows(df):
    """
    Удаляем явные ошибки из датасета: возраст 0 и больше 100.
    """
    res = df.drop(df[(df['age'] <= 0) | (df['age'] > 100)].index)
    return res


def clean_dataset(df):
    res = _drop_na_rows(df)
    res = _drop_invalid_age_rows(res)

    # в df при загрузке из файла могли быть пропуски в колонке 'date_analyse',
    # и тип колонки оказаться object, а не date
    res['date_analyse'] = res['date_analyse'].astype('datetime64[ns]')
    
    return res



def change_columns_type_in_df(df, from_dtype: str, to_dtype: str):
    """
    Заменяет типы данных в колонках датафрейма.
    Если не указаны колонки, то меняет типы во всех колонках.
    """
    res_df = df.copy()
    
    columns = res_df.select_dtypes(include=from_dtype).columns      
    res_df[columns] = res_df[columns].astype(to_dtype)

    return res_df
