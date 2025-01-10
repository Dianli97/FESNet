import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from semg_spike_regression.dataset import ninaprodb8 as db8
from semg_spike_regression.cochlear.bands import filter_band_and_rectify
import matplotlib.pyplot as plt

# min_val = -94.46878814697266
# max_val = 138.81414794921875

def create_sliding_windows(x_bp, y_doa, window_size, step):
    x_windows = [x_bp[:, i:i + window_size] for i in range(0, x_bp.shape[1] - window_size + 1, step)]
    y_windows = [y_doa[:, i:i + window_size] for i in range(0, y_doa.shape[1] - window_size + 1, step)]
    
    return np.array(x_windows), np.array(y_windows)


def preprocess_data(path, subj_list, ex_list, acq_list, window_size):
    x_windows_list = []
    y_windows_list = []
    for idx_subj, idx_ex, idx_acq in product(subj_list, ex_list, acq_list):
        print(f"\n\n\n SUBJECT {1 + idx_subj}/{db8.NUM_SUBJECTS}, EXERCISE {1 + idx_ex}/{db8.NUM_EXERCISES}, ACQUISITION {1 + idx_acq}/{db8.NUM_ACQUISITIONS} \n\n\n")

        # Load the original released raw data
        x_raw, y_doa = db8.load_downloaded_session(path, idx_subj, idx_ex, idx_acq, verbose=True)
        # print(f'Original labels - Min: {y_doa.min().item()}, Max: {y_doa.max().item()}')
        # y_doa = (y_doa - min_val) / (max_val - min_val)
        x_raw = x_raw * 10000

        # Band-pass filtering and rectification
        x_bp = filter_band_and_rectify(
            x=x_raw,
            f_hz=db8.FS_HZ,
            lowcut_hz=20.0,
            highcut_hz=450.0,
            order=4,
            bandplot=False,
        )


        x_windows, y_windows = create_sliding_windows(x_bp, y_doa, window_size, step=window_size)

        
        if x_windows.shape[2] == window_size:
            x_windows_list.extend(x_windows)
            y_windows_list.extend(y_windows)

    x_windows_array = np.array(x_windows_list)
    y_windows_array = np.array(y_windows_list)
  
    return x_windows_array, y_windows_array

