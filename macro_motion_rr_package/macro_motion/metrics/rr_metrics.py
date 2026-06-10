import numpy as np

def compute_regression_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mape = np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0
    bias = np.mean(err)
    sd_err = np.std(err)
    r = np.corrcoef(y_true, y_pred)[0,1] if len(y_true) > 1 else np.nan
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - ss_res / (ss_tot + eps)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_percent": float(mape),
            "Bias": float(bias), "SD_error": float(sd_err),
            "Pearson_r": float(r), "R2": float(r2), "N": int(len(y_true))}
