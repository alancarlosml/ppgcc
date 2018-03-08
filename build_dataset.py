from keras.preprocessing import image
import os
import glob
import numpy as np
from PIL import Image

def get_folders(data_base):
  
  data_folders = []
  for name in os.listdir(data_base):
    if(os.path.isdir(data_base + name)):
      data_folders.append(name)
  print(data_folders)

  return data_folders

def sample(iterable, n, mapa):

  it = iterable
  iterable = []
  for i in range(0,len(it)-1):
    if(i not in mapa):
      iterable.append(it[i])

  reservoir = []
  for t, item in enumerate(iterable):
    if t < n:
      reservoir.append(item)
      mapa.append(t)
    else:
      m = np.random.randint(0,t)
      if m < n:
        reservoir[m] = item
        mapa[m] = t

  return reservoir

def config_base(folders, data_base, train_perc, test_perc, img_type):
  
  train_data = []
  test_data = []
  valid_data = []
  for f in folders:
    dataset = glob.glob(data_base + f + "/*." + img_type)
    datasize = len(dataset)
    train_num = int(datasize*train_perc)
    test_num = int(datasize*test_perc)
    valid_num = 0
    if(train_perc + test_perc < 1.0):
      valid_num = datasize - train_num - test_num #int(datasize*valid_perc)

    print("In folder " + f + ": " + str(datasize) + " images found.")
    print("Train data: " + str(train_num))
    print("Test data: " + str(test_num))
    print("Valid data: " + str(valid_num))

    used = []
    train = sample(dataset, train_num, used)
    train_data.append(train)
    # print(len(used))

    test = sample(dataset, test_num, used)
    test_data.append(test)
    # print(len(used))

    if(valid_num>0):
      valid = sample(dataset, valid_num, used)
      valid_data.append(valid)
      # print(len(used))

  return train_data, test_data, valid_data

def resize_centered(img):
  
  w = img.size[0]; h = img.size[1]; c = len(img.split())
  maior = w
  if(h > maior): 
    maior = h

  if(maior % 2 == 1):
    maior += 1

  n = maior

  old_size = img.size
  new_size = (n, n)
  new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
  new_im.paste(img, ((new_size[0]-old_size[0])/2, (new_size[1]-old_size[1])/2))

  return new_im

def load_data(base_set, image_size=(32 ,32), channels=3):

  x = []
  y = []

  classe = 0
  for class_set in base_set:
    for img_path in class_set:
      img = image.load_img(img_path, grayscale=False, target_size=None)
      res = image.img_to_array(img)

      x.append(res)
      y.append(classe)

    classe += 1

  x = np.asarray(x)
  y = np.asarray(y)
  y = y.reshape(y.shape[0], 1)

  return x, y
  
 
def load(data_root='C:\\Mestrado\\Data\\', img_rows=32, img_cols=32, channels=3, img_type="jpg"):
  
  # Getting folders in data dabe
  classes_folders = get_folders(data_root)

  # Separating base into train and test sets
  train_set, test_set, valid_set = config_base(classes_folders, data_root, train_perc=0.5, test_perc=0.3, img_type=img_type)

  # Loading data
  print("Loading data...")
  (x_train, y_train) = load_data(train_set, (img_rows, img_cols), channels)
  (x_test, y_test) = load_data(test_set, (img_rows, img_cols), channels)
  (x_valid, y_valid) = load_data(valid_set, (img_rows, img_cols), channels)

  return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)
