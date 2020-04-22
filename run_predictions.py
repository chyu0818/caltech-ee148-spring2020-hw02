import os
import numpy as np
import json
from PIL import Image

THRESHOLD_SQ = 0.93
THRESHOLD_RECT = 0.9

def find_red_light_info(red_light, mult):
    # convert to numpy array
    red_light_arr = np.asarray(red_light)
    (height, width, n_chanels) = np.shape(red_light_arr)

    # flatten
    red_light_flat = np.reshape(red_light_arr, np.size(red_light_arr))

    # (left, top, right, down) distance from center to each edge
    center = [4, 4, 4, 4]
    # adjust accordingly to resize
    center = [int(mult*x) for x in center]
    center[2] = width - center[0]
    center[3] = height - center[1]
    print('m', mult, center)
    return {'img': red_light_arr,
            'flat': red_light_flat,
            'norm': np.linalg.norm(red_light_flat),
            'center': center}

def find_target_red_lights():
    red_light_lst = [] # list of red lights of different sizes
    # Open file to get target red light.
    fn = 'RL-001.jpg'
    img = Image.open(os.path.join(data_path, fn))

    # Find square red light.
    (left, top, right, bottom) = (316, 154, 323, 162)
    (width, height) = (right-left, bottom-top)
    red_light = img.crop((left, top, right, bottom))
    red_light.save('red_light.jpg')

    # Find full traffic light (rectangle).
    (left_rec, top_rec, right_rec, bottom_rec) = (316, 154, 323, 172)
    (width_rec, height_rec) = (right_rec-left_rec, bottom_rec-top_rec)
    red_light_rec = img.crop((left_rec, top_rec, right_rec, bottom_rec))
    red_light_rec.save('red_light_rec.jpg')

    red_light_lst.append({'sq':find_red_light_info(red_light, 1),
                          'rec':find_red_light_info(red_light_rec, 1)})
    print('m', 1, ',Sq:', red_light.size, 'Rec', red_light_rec.size)

    # Different sizes for red light.
    mults = [1.5, 2, 2.5, 3, 3.5]
    for m in mults:
        # Resize images (sq and rec) and save for reference.
        red_light_img = red_light.resize((int(m*width), int(m*height)))
        red_light_img.save('red_light' + str(m) + '.jpg')

        red_light_img_rec = red_light_rec.resize((int(m*width_rec), int(m*height_rec)))
        red_light_img_rec.save('red_light_rec' + str(m) + '.jpg')

        red_light_lst.append({'sq':find_red_light_info(red_light_img, m),
                              'rec':find_red_light_info(red_light_img_rec, m)})
        print('m', m, ',Sq:', red_light_img.size, 'Rec:', red_light_img_rec.size)
    return red_light_lst

def compute_convolution_grid(heatmap, row, col, I, T_width, T_flat, T_norm, T_center, threshold):
    (n_rows,n_cols,n_channels) = np.shape(I)

    # Extract template info.
    [left, top, right, bottom] = T_center

    # Find indices of bounding box.
    tl_row = int(row-top)
    br_row = int(row+bottom)
    tl_col = int(col-left)
    br_col = int(col+right)
    # Check that indices are within range.
    if tl_row >= 0 and br_row < n_rows and tl_col >= 0 and br_col < n_cols:
        # Define bounding box
        I_box = I[tl_row:br_row,tl_col:br_col,:]

        # Flatten image within bounding box and find norm.
        I_box_flat = np.float64(np.reshape(I_box, np.size(I_box)))
        I_box_norm = np.linalg.norm(I_box_flat)

        # Find inner product and divide by product of norms.
        comp_val = np.dot(T_flat, I_box_flat) / (T_norm * I_box_norm)

        # If value meets threshold, return value and bounding box.
        if comp_val > threshold and comp_val > heatmap[row,col,0]:
            heatmap[row,col,0] = comp_val
            heatmap[row,col,1] = T_width
    return heatmap

def compute_convolution(I, T, T_center, inds, heatmap, threshold=0.9, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays)
    and returns a heatmap where each grid represents the output produced by
    convolution at each location. You can add optional parameters (e.g. stride,
    window_size, padding) to create additional functionality.
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    # Find properties of template.
    (T_height,T_width,info) = np.shape(T)
    T_flat = np.reshape(T, np.size(T))
    T_norm = np.linalg.norm(T_flat)

    '''
    BEGIN YOUR CODE
    '''

    for [r,c] in inds:
        heatmap = compute_convolution_grid(heatmap, r, c, I, T_width, T_flat, T_norm, T_center, threshold)

    '''
    END YOUR CODE
    '''

    return heatmap

def check_overlap(box1, box2):
    [tl_row1, tl_col1, br_row1, br_col1] = box1
    [tl_row2, tl_col2, br_row2, br_col2] = box2
    # check if share side

    if (tl_col1 >= tl_col2 and tl_col1 <= br_col2 \
      and tl_row1 >= tl_row2 and tl_row1 <= br_row2) \
      or (tl_col1 >= tl_col2 and tl_col1 <= br_col2 \
      and br_row1 >= tl_row2 and br_row1 <= br_row2) \
      or (br_col1 >= tl_col2 and br_col1 <= br_col2 \
      and tl_row1 >= tl_row2 and tl_row1 <= br_row2) \
      or (br_col1 >= tl_col2 and br_col1 <= br_col2 \
      and br_row1 >= tl_row2 and br_row1 <= br_row2):
        return True
    else:
        return False



def remove_overlaps(box, output):
    coords = box[0:4]
    score = box[4]
    # Iterate over all boxes currently in output.

    is_max = True
    k = 0
    while k < len(output):
        coords1 = output[k][0:4]
        score1 = output[k][4]
        # Check for overlapping boxes.
        if check_overlap(coords, coords1) or check_overlap(coords1, coords):
            # Remove box with smaller value.
            if score >= score1:
                output.pop(k)
            else:
                # could also just return after appending this to end
                is_max = False
                break
        else:
            k += 1
    if is_max:
        output.append(box)
    return output

    # Convert list of tuples (inds, val) to just list of inds.
    bounding_boxes = [inds for (inds, fit) in boxes]
    return bounding_boxes

def make_gradient_box(size, range_min=0.5, range_max=1.):
    # always odd
    # range from 0.6 to 1
    assert(size % 2 == 1)
    levels = 1 + int(size / 8)
    if levels == 1:
        levels_incr = 0
    else:
        levels_incr = (range_max - range_min) / (levels - 1)

    box = np.zeros((size, size))
    center = int(size // 2)
    for rad in range(levels):
        curr_val = range_max - levels_incr * rad
        box[center-rad,center-rad:center+rad+1] = curr_val
        box[center+rad,center-rad:center+rad+1] = curr_val
        box[center-rad+1:center+rad,center-rad] = curr_val
        box[center-rad+1:center+rad,center+rad] = curr_val
    return box

def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''

    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''
    (height, width, info) = np.shape(heatmap)
    for r in range(height):
        for c in range(width):
            if heatmap[r,c,0] > 0:
                box_size_half = int(heatmap[r,c,1] / 2)
                [tl_row, tl_col, br_row, br_col] = [r - box_size_half, c - box_size_half, r + box_size_half, c + box_size_half]
                if tl_row > 0 and tl_col > 0 and br_row < height and br_col < width:
                    grad_box = make_gradient_box(box_size_half * 2 + 1)
                    score = np.sum((np.multiply(grad_box, heatmap[tl_row:br_row+1,tl_col:br_col+1,0])) / np.sum(grad_box))
                    print(score)
                    output = remove_overlaps([tl_row, tl_col, br_row, br_col, score], output)

        # output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''

    return output

def detect_red_light_mf(I, T_lst):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>.
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>.
    The first four entries are four integers specifying a bounding box
    (the row and column index of the top left corner and the row and column
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1.

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''

    (n_rows,n_cols,n_channels) = np.shape(I)
    heatmap = np.zeros((n_rows,n_cols,2))

    # heatmap_prev = np.zeros((n_rows,n_cols,2))
    # # Prioritize very bright pixels (focus on red).
    # heatmap_prev[I[:,:,0] > 200,0] = 1
    # # Ignore bottom half of picture.
    # heatmap_prev[n_rows//2:,:,0] = 0
    light_inds = np.argwhere(I[:n_rows//2,:,0] > 200)

    for T in T_lst:
        # Find possible locations where the light could be.
        heatmap_rect = compute_convolution(I, T['rec']['img'], T['rec']['center'], light_inds, heatmap, THRESHOLD_RECT)
        # Confirm where the traffic lights are.
        heatmap = compute_convolution(I, T['sq']['img'], T['sq']['center'], np.argwhere(heatmap_rect[:,:,0] > 0), heatmap_rect, THRESHOLD_SQ)


        # Find the maximum of this heatmap and the previous.
        # heatmap = np.zeros((n_rows,n_cols,2))
        # heatmap[:,:,0] = np.maximum(heatmap0[:,:,0], heatmap_sq[:,:,0])
        # print(np.sum(heatmap_sq))
        # print(np.sum(heatmap0))
        # # Track which grids were updated.
        # updated_inds = np.argwhere(heatmap0[:,:,0] != heatmap[:,:,0])
        # print(updated_inds)
        # heatmap[heatmap0[:,:,0] != heatmap[:,:,0],1] = np.shape(T['sq']['img'])[0]

    output = predict_boxes(heatmap)

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/RedLights2011_Medium'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
T_lst = find_target_red_lights()
# for i in range(len(file_names_train)):
for i in range(20,30):
    print(file_names_train[i])

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I, T_lst)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I, T_lst)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
