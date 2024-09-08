import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from semg_spike_regression.dataset import ninaprodb8 as db8
from semg_spike_regression.cochlear.bands import filter_band_and_rectify
import matplotlib.pyplot as plt

# min_val = -94.46878814697266
# max_val = 138.81414794921875

def create_sliding_windows(x_bp, y_doa, window_size, step):
    # 使用滑动窗口法分割数据
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


# 重构之后画像不一，可能数据开窗时排序出了问题
# 定义需要处理的主题列表、练习列表和采集列表
# subj_list = [0]  # 假设你有3个主题
# ex_list = [0]       # 假设每个主题有2种练习
# acq_list = [0, 1]   # 假设每种练习有3次采集

# # 定义窗口大小
# window_size = 200
# test_path = '/home/ld/python/ori_data/s22/'


# # 调用 preprocess_data 函数
# x_windows_array, y_windows_array = preprocess_data(test_path, subj_list, ex_list, acq_list, window_size)

# # 输出结果的形状来验证
# print("X windows shape:", x_windows_array.shape)
# print("Y windows shape:", y_windows_array.shape)

# for idx_subj, idx_ex, idx_acq in product(subj_list, ex_list, acq_list):
#         print(f"\n\n\n SUBJECT {1 + idx_subj}/{db8.NUM_SUBJECTS}, EXERCISE {1 + idx_ex}/{db8.NUM_EXERCISES}, ACQUISITION {1 + idx_acq}/{db8.NUM_ACQUISITIONS} \n\n\n")

#         # Load the original released raw data
#         x_raw, y_doa = db8.load_downloaded_session(test_path, idx_subj, idx_ex, idx_acq, verbose=True)
        
#         x_raw = x_raw * 10000

#         fig, ax = plt.subplots(2, 1)

#         # 使用ax数组中的元素来绘图
#         ax[0].plot(x_raw[0])  # 在第一个子图上绘制
#         ax[0].set_title('x_raw Channel 0')
#         ax[0].set_xlabel('Sample Index')
#         ax[0].set_ylabel('Amplitude')

#         # 使用ax数组中的元素来绘图
#         ax[1].plot(y_doa[0])  # 在第一个子图上绘制
#         ax[1].set_title('y_doa Channel 0')
#         ax[1].set_xlabel('Sample Index')
#         ax[1].set_ylabel('Amplitude')

#         plt.tight_layout()
#         plt.show()
