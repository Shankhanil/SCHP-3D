import cv2
from tqdm import tqdm 
import numpy as np
from glob import glob
import json

mask_dir = sorted(glob("data/mask/*"))

with open('classmaping.json') as json_file:
    labelmaps = json.load(json_file)
    
for m_im  in tqdm(mask_dir):
    mask = cv2.imread(m_im, 0)
    mask_arr = np.ones(mask.shape, dtype=np.uint8)

    for i in labelmaps:
        mask_arr[mask == int(i)] = labelmaps[i]

    pth = "data2/mask/{}".format(m_im.split('/')[-1])
    cv2.imwrite(pth, mask_arr)

