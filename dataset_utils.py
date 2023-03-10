import os
import h5py
import json
import numpy as np


def read_gt_COHFACE(root_dir, subject, record):
    rw_path = os.path.join(root_dir, subject, record, "data.hdf5")

    with h5py.File(rw_path, "r") as f:
        # List all groups
        gt_keys = list(f.keys())
        gt_rr = list(f[gt_keys[1]])
        gt_ts = list(f[gt_keys[2]])  # in seconds

    breaths_file = open(os.path.join(root_dir, "COHFACE_breaths.json"), "r")
    data = json.load(breaths_file)
    breaths_file.close()

    breaths = data[str(subject)][int(record)][str(record)]

    return gt_rr, gt_ts, breaths



def breaths2RR(peaks, window_size, step_size, rw_length, fs):
    window_size *= fs  # multiply by the frame rate of the respiratory belt
    step_size *= fs

    peaks = np.array(peaks)

    start = 0
    RR = []

    while start + window_size < rw_length:
        selected_peaks = peaks[np.where(peaks >= start)]
        selected_peaks = selected_peaks[np.where(selected_peaks <= start + window_size)]

        IBI = np.mean([selected_peaks[i + 1] - selected_peaks[i] for i in range(len(selected_peaks) - 1)]) / fs

        bpm = 60 / IBI

        RR.append(bpm)

        start += step_size

    return RR
