import numpy as np


def rmse(predictions, targets):
    """
    :param predictions: array with predictions
    :param targets: array with ground truth data (should be the same length as the predictions array)

    :return: RMSE value

    Computes Root Mean Square Error (RMSE) for a given predictions and ground truths.
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def mae(predictions, targets):
    """
    :param predictions: array with predictions
    :param targets: array with ground truth data (should be the same length as the predictions array)

    :return: MAE value

    Computes Mean Absolute Error (MAE) for a given predictions and ground truths.
    """
    return abs(predictions - targets).mean()


def success_rate(predictions, targets, bound=2):
    """
    Gets predicted and ground truth value, and checks if predicted lies in
    boundaries +- bound of gt value. For now, boundaries are +- 2 RR points
    Returns percent of values that were predicted successfully (lying within bounds)
    """
    is_success_pred = []
    for gt, pred in zip(targets, predictions):
        if abs(gt - int(pred)) <= bound:
            is_success_pred.append(True)
        else:
            is_success_pred.append(False)
    success_rate = (sum(is_success_pred) / len(is_success_pred)) * 100
    return success_rate



def compute_metrics(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)

    low_gt_indx = np.argwhere(targets <= 6).flatten()
    nan_gt_indx = np.argwhere(np.isnan(targets)).flatten()

    indx_to_remove = list(set(low_gt_indx) | set(nan_gt_indx))

    predictions = np.delete(predictions, indx_to_remove)
    targets = np.delete(targets, indx_to_remove)

    if len(predictions) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    rmse_value = rmse(predictions, targets)
    mae_value = mae(predictions, targets)

    SR_1 = success_rate(predictions, targets, bound=1)
    SR_2 = success_rate(predictions, targets, bound=2)
    SR_3 = success_rate(predictions, targets, bound=3)


    return rmse_value, mae_value, SR_1, SR_2, SR_3

