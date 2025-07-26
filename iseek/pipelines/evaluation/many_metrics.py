import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.morphology import skeletonize


def get_tp_fp_fn_tn(y_true, y_pred, labels=None):
    if labels is None:
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.unique(y_true))
    else:
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=labels)
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    tn = np.sum(cm) - (fp + fn + tp)
    return tp, fp, fn, tn


def sensitivity(y_true, y_pred, exclude_background=True, labels=None):
    tp, fp, fn, tn = get_tp_fp_fn_tn(y_true, y_pred, labels=labels)
    sen = tp / (tp + fn)
    if exclude_background:
        return np.mean(sen[1:])
    else:
        return np.mean(sen)


def sensitivity_v2(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1, 2])
    sensitivities = []
    for i in range(1, 3):
        TP = cm[i - 1, i - 1]
        FN = np.sum(cm[i - 1, :]) - TP
        sen = TP / (TP + FN)  # if TP + FN != 0 else 0
        sensitivities.append(sen)
    return np.mean(sensitivities)


def specificity(y_true, y_pred, exclude_background=True, labels=None):
    tp, fp, fn, tn = get_tp_fp_fn_tn(y_true, y_pred, labels=labels)
    spe = tn / (tn + fp)
    if exclude_background:
        return np.mean(spe[1:])
    else:
        return np.mean(spe)


def accuracy(y_true, y_pred, exclude_background=True, labels=None):
    tp, fp, fn, tn = get_tp_fp_fn_tn(y_true, y_pred, labels=labels)
    acc = (tp + tn) / (tp + tn + fp + fn)
    if exclude_background:
        return np.mean(acc[1:])
    else:
        return np.mean(acc)


def dice_coefficient(y_true, y_pred, exclude_background=True, labels=None):
    tp, fp, fn, tn = get_tp_fp_fn_tn(y_true, y_pred, labels=labels)
    dice = 2 * tp / (2 * tp + fp + fn)
    if exclude_background:
        return np.mean(dice[1:])
    else:
        return np.mean(dice)


def cl_score(v, s):
    return np.sum(v * s) / np.sum(s)


def clDice_single_class(v_p, v_l):
    if len(v_p.shape) == 2:
        tprec = cl_score(v_p, skeletonize(v_l))
        tsens = cl_score(v_l, skeletonize(v_p))
        return 2 * tprec * tsens / (tprec + tsens)
    elif len(v_p.shape) == 3:
        raise NotImplementedError


def calc_clDice_multiclass(y_true, y_pred, num_classes, exclude_background=True):
    cl_dice = np.zeros(num_classes)

    for c in range(num_classes):
        if exclude_background and c == 0:
            continue
        y_true_c = (y_true == c).astype(bool)
        y_pred_c = (y_pred == c).astype(bool)

        cl_dice[c] = clDice_single_class(y_pred_c, y_true_c)

    if exclude_background:
        return np.mean(cl_dice[1:])
    else:
        return np.mean(cl_dice)
