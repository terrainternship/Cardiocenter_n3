from pathlib import Path

import dill
import numpy as np
import pandas as pd

# отключаем печать некоторых предупреждений
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)


ROOT_PATH = Path('.')  # путь к корню проекта


# Загружаем модели

with open(ROOT_PATH / 'models/datapipes/main_fittedon_train_1.dill', 'rb') as f:
    datapipe = dill.load(f)

with open(ROOT_PATH / 'models/predictors/gama_ASH_600s_2_fittedon_train_3.dill', 'rb') as f:
    model = dill.load(f)

# Предсказываем на новых данных

X = pd.read_csv(ROOT_PATH / 'data/raw/example.csv')

X_for_model = datapipe.transform(X)

y_pred = model.predict(X_for_model)
y_pred_rounded = np.round(y_pred, 3)

print('Предсказания:')
print(repr(y_pred_rounded))
