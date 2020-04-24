import os
import json
import numpy as np
import matplotlib.pyplot as plt

def find_overlap(box1, box2):
    [tl_row1, tl_col1, br_row1, br_col1] = box1
    [tl_row2, tl_col2, br_row2, br_col2] = box2

    # Define coordinates of intersection.
    [tl_row, tl_col, br_row, br_col] = [-1,-1,-1,-1]

    if (tl_row1 >= tl_row2 and tl_row1 <= br_row2 \
    and tl_col1 >= tl_col2 and tl_col1 <= br_col2) \
    or (br_row1 >= tl_row2 and br_row1 <= br_row2 \
    and br_col1 >= tl_col2 and br_col1 <= br_col2) \
    or (tl_row1 >= tl_row2 and tl_row1 <= br_row2 \
    and br_col1 >= tl_col2 and br_col1 <= br_col2) \
    or (br_row1 >= tl_row2 and br_row1 <= br_row2 \
    and tl_col1 >= tl_col2 and tl_col1 <= br_col2):
        tl_row = max(tl_row1, tl_row2)
        tl_col = max(tl_col1, tl_col2)
        br_row = min(br_row1, br_row2)
        br_col = min(br_col1, br_col2)
    return [tl_row, tl_col, br_row, br_col]

def find_area(box):
    [tl_row, tl_col, br_row, br_col] = box
    return np.abs(br_row - tl_row) * np.abs(br_col - tl_col)

def compute_iou(box1, box2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    # Determine whether there's overlap.
    box_intersect1 = find_overlap(box1, box2)
    box_intersect2 = find_overlap(box2, box1)
    if box_intersect1 != [-1,-1,-1,-1]:
        box_intersect = box_intersect1
    elif box_intersect2 != [-1,-1,-1,-1]:
        box_intersect = box_intersect2
    else:
        return 0

    # Calculate IOU.
    area_intersect = find_area(box_intersect)
    area_union = find_area(box1) + find_area(box2) - area_intersect
    iou = area_intersect / area_union

    assert (iou >= 0) and (iou <= 1.0)
    return iou

def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.)
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives.
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for pred_file, pred0 in preds.items():
        TP0 = 0
        FP0 = 0
        FN0 = 0

        pred = pred0.copy()
        # Remove values with too low of confidence.
        k = 0
        while k < len(pred):
            if pred[k][4] < conf_thr:
                pred.pop(k)
            else:
                k += 1
        gt = gts[pred_file]
        gt_len = len(gt)
        pred_len = len(pred)

        # Iterate through ground truth.
        for i in range(len(gt)):
            max_ind = -1
            max_fit = iou_thr
            # Iterate through predictions.
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou >= max_fit:
                    max_ind = j
                    max_fit = iou
            # True positive
            if max_ind != -1:
                TP0 += 1
                pred.pop(max_ind)
        # Calculate false positives and false negatives.
        FP0 = pred_len - TP0
        FN0 = gt_len - TP0
        assert(FP0 + TP0 + FN0 == gt_len + pred_len - TP0)

        TP += TP0
        FP += FP0
        FN += FN0

    return TP, FP, FN

def plot_PR(preds, gts):
    # For a fixed IoU threshold, vary the confidence thresholds.
    fig, ax = plt.subplots()

    # Using (ascending) list of confidence scores as thresholds
    confidence_thrs_lst = [preds[fname][i][4] for fname in preds for i in range(len(preds[fname]))]
    confidence_thrs = np.sort(np.array(confidence_thrs_lst,dtype=float))

    colors = ['r', 'b', 'g']
    iou_thrs = [0.25, 0.5, 0.75]
    for j in range(3):
        iou_thr = iou_thrs[j]
        color = colors[j]
        tp = np.zeros(len(confidence_thrs))
        fp = np.zeros(len(confidence_thrs))
        fn = np.zeros(len(confidence_thrs))

        # Calculate for different confidence thresholds.
        for i, conf_thr in enumerate(confidence_thrs):
            tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=iou_thr, conf_thr=conf_thr)

        # Plot training set PR curves
        P = tp / (fp + tp)
        R = tp / (fn + tp)
        ax.plot(R, P, color=color, marker='.', label=str(iou_thr))

    ax.set(xlabel='Recall', ylabel='Precision', title='PR Curve')
    ax.legend(title='IOU threshold')
    plt.show()
    return 0

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Load training data.
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)

with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:

    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)

    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)

# Plot train curves.
print('Plotting train set PR curves.')
plot_PR(preds_train, gts_train)

if done_tweaking:
    # Plot test curves.
    print('Plotting test set PR curves.')
    plot_PR(preds_test, gts_test)
