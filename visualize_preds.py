import json
import numpy as np
from PIL import Image, ImageDraw
import os

def draw_boxes(I, bounding_boxes):
    # iterate through all boxes
    for [tl_row, tl_col, br_row, br_col, score] in bounding_boxes:
        draw = ImageDraw.Draw(I)
        draw.rectangle([tl_col, tl_row, br_col, br_row], outline=(36, 248, 229))
        del draw
    return I

def main():
    # set the path to the downloaded data:
    data_path = '../data/RedLights2011_Medium'

    # set a path for saving predictions:
    preds_path = '../data/hw02_preds'

    # load splits:
    split_path = '../data/hw02_splits'
    file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
    file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))
    file_names = file_names_train

    # get predictions
    # with open(os.path.join(preds_path,'preds.json')) as f:
    with open(os.path.join(preds_path, 'preds_train.json')) as f:
        bounding_boxes = json.load(f)

    # for i in range(len(file_names)):
    for i in range(20,30):
        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names[i]))
        # draw box and save
        I = draw_boxes(I, bounding_boxes[file_names[i]])
        I.save(os.path.join(preds_path,file_names[i]))
    return 0

if __name__ == '__main__':
    main()
