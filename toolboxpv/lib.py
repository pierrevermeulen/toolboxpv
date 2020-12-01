# -*- coding: UTF-8 -*-

""" Main lib for toolboxpv Project
"""
from os.path import split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, \
        mean_squared_error, mean_absolute_error

def plot_history(history, metric, n_epochs=0):
    fig, ax = plt.subplots(1, 2, figsize=(16, 4))
    ax[0].plot(history.history['loss'][n_epochs:], label = 'Train')
    if val_loss in history.history.keys():
        ax[0].plot(history.history['val_loss'][n_epochs:], label = 'Validation')
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(loc='best')

    dict_leg = {'mae': 'Absolute Error',
                'mse': 'Absolute Square Error',
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'f1': 'F1 score'}
    ax[1].plot(history.history[metric][n_epochs:], label = 'Train')
    if 'val_'+metric in history.history.keys():
        ax[1].plot(history.history['val_'+metric][n_epochs:], label = 'Validation')
    ax[1].set_title('Model mean '+ dict_leg[metric])
    ax[1].set_ylabel('Mean '+ dict_leg[metric])
    ax[1].set_xlabel('Epoch')
    ax[1].legend(loc='best')
    plt.show()

    return ax

def show_results(y_test, y_pred, metric, label = ''):
    if metric == 'accuracy':
        score = accuracy_score(y_test, y_pred)
    elif metric == 'precision':
        score = precision_score(y_test, y_pred)
    elif metric == 'F1':
        score = f1_score(y_test, y_pred)
    elif metric == 'mae':
        score = mean_absolute_error(y_test, y_pred)
    elif metric == 'mean_squared_error':
        score = mean_squared_error(y_test, y_pred)

    print(f"Final {label} {metric}: {score:.4f}")
    return score

if __name__ == '__main__':
    # For introspections purpose to quickly get this functions on ipython
    import toolboxpv

