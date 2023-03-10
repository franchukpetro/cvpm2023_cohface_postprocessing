import os

from dataset_utils import read_gt_COHFACE, breaths2RR
from metrics import compute_metrics

WINDOW_SIZE=20
STEP_SIZE = 1


def test_cohface():
    root_dir = "/path/to/COHFACE/root/folder/"

    records2skip = ["4_2", "26_1", "26_2", "26_3", "31_3", "36_0", "36_2", "36_3", "37_0", "37_1", "37_2", "37_3"]

    for r in range(4):
        for s in range(1, 41):
            s, r = str(s), str(r)
            record_name = s + "_" + r

            video_path = os.path.join(root_dir, s, r, "data.avi")

            # skip records with low or bad GT
            if record_name in records2skip:
                continue

            ##########

            # RR algorithm predicts RR values
            RR_values = []

            ##########

            GT_RW, GT_RW_ts, peaks = read_gt_COHFACE(root_dir, str(s), str(r))

            GT_values = breaths2RR(peaks, window_size=WINDOW_SIZE, step_size=STEP_SIZE, rw_length=len(GT_RW), fs=256)
            GT_ts = [WINDOW_SIZE + i for i in range(len(GT_values))]

            if len(RR_values) > len(GT_values):
                RR_values = RR_values[:len(GT_values)]
            elif len(RR_values) < len(GT_values):
                GT_values = GT_values[:len(RR_values)]


            rmse, mae, SR_1, SR_2, SR_3 = compute_metrics(RR_values, GT_values)
