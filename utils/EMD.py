import PyEMD
from sklearn.linear_model import Lasso
from scipy.signal import savgol_filter
import numpy as np


def find_freqs(emd_x, data_x, start_loc, end_loc):
    current_data = data_x.reshape(-1)
    current_data = savgol_filter(current_data, 5, 2)
    current_len = end_loc - start_loc
    C_IMF = emd_x.emd(current_data)

    g_freq = []
    g_Amp = []
    for c_n, c_imf in enumerate(C_IMF):
        if c_n != len(C_IMF) - 1:
            CF_imf = np.fft.fft(c_imf)
            A_CF_imf = np.abs(CF_imf)
            cmax_freq_group = np.argsort(A_CF_imf[: current_len // 2 + 1])

            c_j = 0
            while cmax_freq_group[-(c_j + 1)] <= 1:  # Ignoring too low frequency or trend
                c_j += 1
            if A_CF_imf[-(c_j + 1)] < 0.2:  # Ignoring too small amplitude
                continue
            c_freq_length = cmax_freq_group[-(c_j + 1)]
            if c_freq_length in g_freq:
                g_Amp[g_freq.index(c_freq_length)] += A_CF_imf[-(c_j + 1)]
            else:
                g_freq.append(c_freq_length)
                g_Amp.append(A_CF_imf[-(c_j + 1)])
    if len(g_Amp) != 0:
        s_Amp = max(g_Amp)
        s_freq = g_freq[g_Amp.index(s_Amp)]
    else:
        s_freq = 0
    return g_freq, s_freq


def EMD_Find_Freq(df_values, seq_length):
    emd = PyEMD.EMD(std_thr=0.01, range_thr=0.05)
    local_freq, s_freq = find_freqs(emd, df_values, 0, seq_length)
    local_freq = list(set(local_freq))
    return np.array(local_freq), s_freq


def EMD_Reconstruct(df_values, seq_length, input_features):

    # Define the level
    input_trend = np.ones([1, seq_length])

    if input_features is not None:
        input_features = np.concatenate([input_trend, input_features], axis=0)
    else:
        input_features = input_trend

    lasso_sklearn = Lasso(alpha=0.1, max_iter=1000, fit_intercept=False)  # ETTm, ECL, Traffic, Weather, Solar, Air
    # lasso_sklearn = Lasso(alpha=0.01, max_iter=1000, fit_intercept=False)  # River
    # lasso_sklearn = Lasso(alpha=0.001, max_iter=1000, fit_intercept=False)  # ETTh, BTC, ETH
    lasso_sklearn.fit(input_features.transpose(1, 0), df_values)
    return lasso_sklearn.coef_


def EMD_Predict(coef, input_length, pred_length, local_freq, sin_waves, cos_waves):
    pred_features = []
    sin_features = []
    cos_features = []
    pred_trend = np.ones(input_length + pred_length)
    pred_features.append(pred_trend)

    local_freq = local_freq.reshape(-1)
    local_freq = local_freq.tolist()
    local_freq = list(set(local_freq))
    lasso_sklearn = Lasso()

    coef_num = 1
    for g_p in local_freq:
        if g_p == 0:
            continue
        sin_features.append(sin_waves[int(g_p), :])
        cos_features.append(cos_waves[int(g_p), :])
        coef_num += 2
    lasso_sklearn.coef_ = coef[:coef_num]
    lasso_sklearn.intercept_ = 0

    pred_features.extend(sin_features)
    pred_features.extend(cos_features)
    pred_features = np.array(pred_features).transpose(1, 0)
    prediction_seq = lasso_sklearn.predict(pred_features)

    return prediction_seq
