from scipy.misc import imresize,imsave
from PIL import Image
import os

train_0 = "C:\\Mestrado\\newdata\\0"
train_1 = "C:\\Mestrado\\newdata\\1"

for filename1 in os.listdir(train_0):
	img = Image.open(train_0 + "\\" + filename1)
	new_img = imresize(img, size=(32, 32), interp='cubic')
	imsave('C:\\Mestrado\\newdata\\0\\' + filename1, new_img)
    
for filename2 in os.listdir(train_1):
    img = Image.open(train_1 + "\\" + filename2)
    new_img = imresize(img, size=(32, 32), interp='cubic')
    imsave('C:\\Mestrado\\newdata\\1\\' + filename2, new_img)