from glob import glob 
import random
from sklearn.model_selection import train_test_split
import numpy as np
import shutil
from tqdm import tqdm 

dir_pth = glob("/home/bigthinx/research/Self-Correction-Human-Parsing/data/image/*")

dir_pth = [i.split('.')[0] for i in dir_pth]
print(len(dir_pth))
random.seed(15)
random.shuffle(dir_pth)
data = np.array(dir_pth)

x_train, x_test = train_test_split(data,test_size=0.2)

for i, file in tqdm(enumerate(x_train)):

    im = file + '.jpg'
    msk = file.replace('image', 'mask') + '.png'

    im_sav = "data2/image/train/"  +  str(i).zfill(8) + '.jpg'
    msk_sav = "data2/mask/train/" + str(i).zfill(8) + '.png'
    shutil.copy(im, im_sav)
    shutil.copy(msk, msk_sav)


for i, file in tqdm(enumerate(x_test)):

    im = file + '.jpg'
    msk = file.replace('image', 'mask') + '.png'

    im_sav = "data2/image/test/"  + str(i).zfill(8) + '.jpg'
    msk_sav = "data2/mask/test/" + str(i).zfill(8) + '.png'
    shutil.copy(im, im_sav)
    shutil.copy(msk, msk_sav)


# print(len(x_train), len(x_test))
# print(x_test)

