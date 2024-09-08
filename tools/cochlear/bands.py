"""
    Author(s):
    Marcello Zanghieri <marcello.zanghieri2@unibo.it>
    
    Copyright (C) 2023 University of Bologna and ETH Zurich
    
    Licensed under the GNU Lesser General Public License (LGPL), Version 2.1
    (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
        https://www.gnu.org/licenses/lgpl-2.1.txt
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ssg
import matplotlib.pyplot as plt


ORDER = 4
LOWCUT_HZ = 20.0
HIGHCUT_HZ = 450.0


def create_butterworth_filter(
    f_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
) -> np.ndarray:

    fnyquist_hz = f_hz / 2.0
    lowcut_norm = lowcut_hz / fnyquist_hz
    highcut_norm = highcut_hz / fnyquist_hz

    Wn = [lowcut_norm, highcut_norm]
    btype = 'bandpass'

    sos = ssg.butter(N=order, Wn=Wn, btype=btype, output='sos')

    return sos


def visualize_filter(
    sos: np.ndarray,
    f_hz: float,
    num_freqs: int = 500,
    xscale: str = 'linear',
) -> None:

    w, h = ssg.sosfreqz(sos, worN=num_freqs, fs=f_hz)
    abs_h = np.abs(h)
    abs_db_h = 20.0 * np.log10(abs_h)
    angle_h = np.angle(h)

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(8.0, 8.0)
    )
    fig.suptitle("Frequency response of the filter")
    ax1.set_title("Amplitude (linear)")
    ax2.set_title("Amplitude (logarithmic)")
    ax3.set_title("Phase")

    ax1.set_ylabel("Gain (dimensionless)")
    ax2.set_ylabel("Gain (dB)")
    ax3.set_ylabel("Phase (rad)")
    ax3.set_xlabel("Frequency (Hz)")

    ax1.plot(w, abs_h, label="Amplitude (linear)")
    ax2.plot(w, abs_db_h, label="Amplitude (logarithmic)")
    ax3.plot(w, angle_h, label="Phase")

    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    ax1.set_xscale(xscale)
    ax1.set_xlim([0.0, f_hz / 2.0])
    ax1.set_ylim([0.0, 1.0])
    ax2.set_ylim([-100.0, 0.0])
    ax3.set_ylim([-np.pi, +np.pi])

    fig.tight_layout()
    plt.show()

    return


def filter_band(x: np.ndarray, sos: np.ndarray) -> np.ndarray:

    x_bp = ssg.sosfilt(sos, x, axis=-1)

    return x_bp


def full_rectify(x: np.ndarray) -> np.ndarray:
    return np.abs(x)


def filter_band_and_rectify(
    x: np.ndarray,
    f_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
    bandplot: bool = False,
    bandplot_num_freqs: int = 500,
    bandplot_xscale: str = 'linear',
) -> np.ndarray:

    sos = create_butterworth_filter(
        f_hz=f_hz, lowcut_hz=lowcut_hz, highcut_hz=highcut_hz, order=order,
    )
    if bandplot:
        visualize_filter(
            sos,
            f_hz=f_hz,
            num_freqs=bandplot_num_freqs,
            xscale=bandplot_xscale,
        )

    x = filter_band(x, sos)
    x = full_rectify(x)

    return x


def main():
    pass


if __name__ == '__main__':
    main()
