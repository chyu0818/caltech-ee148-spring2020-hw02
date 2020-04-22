import os
import json
import numpy as np

def find_overlap(box1, box2):
    [tl_row1, tl_col1, br_row1, br_col1] = box1
    [tl_row2, tl_col2, br_row2, br_col2] = box2
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
    return (br_row - tl_row) * (br_col - tl_row)

def compute_iou(box1, box2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    box_intersect1 = find_overlap(box1, box2)
    box_intersect2 = find_overlap(box2, box1)
    if box_intersect1 != [-1,-1,-1,-1]:
        box_intersect = box_intersect1
    elif box_intersect2 != [-1,-1,-1,-1]:
        box_intersect = box_intersect2
    else:
        return 0

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

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.iteritems():
        gt = gts[pred_file]
        orig_len = len(gt) + len(pred)
        # Iterate through ground truth
        for i in range(len(gt)):
            TP0 = 0
            FP0 = 0
            FN0 = 0
            max_ind = -1
            max_fit = iou_thr
            # Iterate through predictions
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if iou > max_fit and pred[j][4] > conf_thr:
                    max_ind = j
                    max_fit = iou
            # True positive
            if max_ind != -1:
                TP0 += 1
                pred.pop(max_ind)
            # False negative
            else:
                FN0 += 1
        FP0 += len(pred)
        assert(FP0 + TP0 + FN0 == orig_len)#temp
        TP = TP0
        FP = FP0
        FN = FN0


    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

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


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.


confidence_thrs = np.sort(np.array([preds_train[fname][4] for fname in preds_train],dtype=float)) # using (ascending) list of confidence scores as thresholds
tp_train = np.zeros(len(confidence_thrs))
fp_train = np.zeros(len(confidence_thrs))
fn_train = np.zeros(len(confidence_thrs))
for i, conf_thr in enumerate(confidence_thrs):
    tp_train[i], fp_train[i], fn_train[i] = compute_counts(preds_train, gts_train, iou_thr=0.5, conf_thr=conf_thr)

# Plot training set PR curves

if done_tweaking:
    print('Code for plotting test set PR curves.')
