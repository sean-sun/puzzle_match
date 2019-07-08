# To find the closest matching region in master to sub, a straightforward way is to convolve master with sub,
# the position which gets the strongest activation should be the center of the closest matching region

import sys
import cv2
import numpy as np

# load the puzzle_master and puzzle_sub image
master = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
sub = cv2.imread(sys.argv[2], cv2.IMREAD_GRAYSCALE)

# resize image: smaller side of master to 400
h, w = master.shape
scale = 400.0 / min(h, w)
master = cv2.resize(master, (int(w*scale), int(h*scale)))
sub = cv2.resize(sub, (int(sub.shape[1]*scale), int(sub.shape[0]*scale)))

master = (master < 128).astype(np.uint8)
sub = (sub < 128).astype(np.uint8)

# dilate puzzle_master so that puzzle_sub with some minor shape mismatch could still fit in
kernel = np.ones((9, 9), np.uint8)
master = cv2.dilate(master, kernel, iterations=1)

# remove the surrounding zero pixel of puzzle_sub
hs = np.sum(sub, axis=1)
ws = np.sum(sub, axis=0)
h_axis = np.where(hs > 0)[0]
w_axis = np.where(ws > 0)[0]
h1, h2 = h_axis[0], h_axis[-1]
w1, w2 = w_axis[0], w_axis[-1]
sub = sub[h1:h2+1, w1:w2+1]

# if we can rotate puzzle_sub to fit in puzzle_master, we should consider it
def gen_filter(sub, angle):
    h, w = sub.shape
    if angle == 0:     ### rotate 0
        filter = sub
    elif angle == 90:  ### rotate 90
        filter = np.zeros((w, h), dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                filter[i, j] = sub[j, w-1-i]
    elif angle == 180: ### rotate 180
        filter = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                filter[i, j] = sub[h-1-i, w-1-j]
    elif angle == 270: ### rotate 270
        filter = np.zeros((w, h), dtype=np.uint8)
        for i in range(w):
            for j in range(h):
                filter[i, j] = sub[h-1-j, i]
    return filter

# match master with sub using template matching
def puzzle_match(master, sub):
    max_xy, max_v, max_p = (-1, -1), -1, -1
    # rotate Puzzle_Sub (4) or not (1)
    for k in range(1):
        filter = gen_filter(sub, 90*k)
        res = cv2.matchTemplate(master, filter, cv2.TM_CCORR_NORMED)
        ind = np.unravel_index(np.argmax(res, axis=None), res.shape)
        if max_v < res[ind]:
            max_v = res[ind]
            max_xy = ind
            max_p = k
    return max_v, max_xy, max_p

max_v, max_xy, max_p = puzzle_match(master, sub)
print max_v, max_xy, max_p

# draw the matching result
y, x = max_xy
filter = gen_filter(sub, max_p*90)
master = cv2.imread(sys.argv[1], 1)
py, px = np.where(filter)
for i in range(px.shape[0]):
    cv2.circle(master, (int((x+px[i])/scale), int((y+py[i])/scale)), 3, (0, 0, 255))
cv2.imwrite('match_res.jpg', master)

