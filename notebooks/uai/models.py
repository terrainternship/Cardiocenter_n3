
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


cmap_gr = LinearSegmentedColormap.from_list('green_red', ['g', 'r'])


def calculate_metrics(y_pred, y_true) -> dict:
    y_pred = getattr(y_pred, 'values', y_pred)
    y_true = getattr(y_true, 'values', y_true)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape_value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    abs_diff = np.abs(y_true - y_pred)
    diff_below_threshold = (abs_diff <= 0.5).sum()
    total_predictions = len(y_pred)
    percentage_diff = (diff_below_threshold / total_predictions) * 100
 
    return dict(mae=mae, mse=mse, r2=r2, mape_value=mape_value, percentage_diff=percentage_diff)


def print_metrics(metrics: dict):
    print("{:<55} {:>5.2f}".format("Средняя абсолютная ошибка (MAE):", metrics['mae']))
    print("{:<55} {:>5.2f}".format("Среднеквадратичная ошибка (MSE):", metrics['mse']))
    print("{:<55} {:>5.2f}".format("Коэффициент детерминации (R^2):", metrics['r2']))
    print("{:<55} {:>5.2f}%".format("Средняя абсолютная процентная ошибка (MAPE):", metrics['mape_value']))
    print("{:<55} {:>5.2f}%".format("Доля предсказаний с отклонением не более 0,5 мг:", metrics['percentage_diff']))


def plot_predictions(y_pred, y_true, title='Сравнение предсказанной и назначенной доз'):
    ae = np.abs(y_pred - y_true)
    # size = 20 * ae + 1

    plt.axis('equal')
    plt.axline((0, 0), slope=1, color='k', alpha=0.1)

    # Добавляем точечную диаграмму
    # plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='lightgray')
    scatter = plt.scatter(y_pred, y_true, marker='.', c=ae, cmap=cmap_gr)
    # scatter.plot([0, 1], [0, 1], transform=scatter.axes.transAxes)

    # Добавляем colorbar
    plt.colorbar(scatter, label='Абсолютная ошибка')

    plt.title(title + f' (N={len(y_pred)})')
    plt.xlabel('Предсказанная доза')
    plt.ylabel('Назначенная доза')
    plt.grid(True)
    plt.show()
