import numpy as np
from thequickmath.exceptions import *

def gradient_descent(df, init_x, init_alpha, improvement_percent_threshold=0.1):
    if len(df) != init_x.shape[0]:
        raise DimensionsDoNotMatch('Number of derivatives does not match x dimension')

    def get_value(f, x):
        values = np.zeros((len(f),))
        for i in range(len(f)):
            values[i] = f[i](x)
        return values

    alpha = init_alpha
    x = init_x
    improvement_percent = 1000
    x_history = []
    while improvement_percent > improvement_percent_threshold:
        df_at_x = get_value(df, x)
        x = x - alpha * df_at_x
        x_history.append(x)
        improvement_percent = np.linalg.norm(np.abs(x_history[-1] - x_history[-2])) / np.linalg.norm(x_history[-2])
        print('\tL2(x) = ' + str(np.linalg.norm(x)) + ', L2(df) = ' + str(np.linalg.norm(df_at_x)) + ', improvement = ' + improvement_percent*100 + '%')
    return x, x_history