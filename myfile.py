import numpy as np

arr = [(1, 2), (3, 4), (5, 6), (7, 8)]

for (first, second) in arr:
    print(second)


'''
(best_w, train_err) = least_squares(x_train, y_train, M_values[0])  # poczatkowa wartosc do porownywania
best_m = 0
for m in M_values:
    (w, err) = least_squares(x_train, y_train, m)
    if err < train_err:
        best_m = m
        best_w = w
        train_err = err
val_err = least_squares(x_val, y_val, best_m)

return (best_w, train_err, val_err)


 wm = []
    errs = []
    for m in M_values:
        wm.append(least_squares(x_train, y_train, m))
    for (w, err) in wm:
        errs.append(mean_squared_error(x_val, y_val, w))
    index = errs.index(min(errs))

    return (wm[index][0], wm[index][1], errs[index])
'''