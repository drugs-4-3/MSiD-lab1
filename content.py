# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from numpy.linalg import inv
from utils import polynomial

def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    differences_vector = np.subtract(y, polynomial(x, w))
    squared_vector = np.multiply(differences_vector, differences_vector)
    sum = np.sum(squared_vector)
    return (1 / x.size) * sum


def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    x_train = x_train[:, 0] # zamiast dac wektor to macierz wektorowa, echh...
    array = [[x ** m for m in range(M + 1)] for x in x_train]
    return np.array(array)



def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    fi = design_matrix(x_train, M)
    w = inv(fi.transpose() @ fi) @ fi.transpose() @ y_train
    return (w, mean_squared_error(x_train, y_train, w))


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''

    fi = design_matrix(x_train, M)
    fi_squared = fi.transpose() @ fi
    size = fi_squared.shape[0]

    w = inv(fi.transpose() @ fi + regularization_lambda * np.identity(size)) @ fi.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)

    return (w, err)



def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''

    w_list = []
    train_errs = []
    val_errs = []
    for m in M_values:
        (w, err) = least_squares(x_train, y_train, m)
        w_list.append(w)
        train_errs.append(err)
        val_errs.append(mean_squared_error(x_val, y_val, w))
    index = val_errs.index(min(val_errs))

    return (w_list[index], train_errs[index], val_errs[index])


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''

    train_errs = []
    val_errs = []
    w_list = []
    for lambd in lambda_values:
        (w, err) = regularized_least_squares(x_train, y_train, M, lambd)
        w_list.append(w)
        train_errs.append(err)
        val_errs.append(mean_squared_error(x_val, y_val, w))
    index = val_errs.index(min(val_errs))

    return (w_list[index], train_errs[index], val_errs[index], lambda_values[index])
