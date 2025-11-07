import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from os import walk
from pprint import pprint
from PIL import Image
import re

import tensorflow as tf
from tensorflow import keras

def addpxl(img, coordinate_pxl, pxl_color):
    color = (255, 0, 0) if pxl_color == 0 else (0, 255, 0)

    pprint(coordinate_pxl)
    img.putpixel(coordinate, color)
    img.save("final_flag.png")


_, _, file_paths = next(walk("Flag"))
model = tf.keras.models.load_model("Model.h5", compile=False)
flag_img = Image.new("RGB",(200,200), (150, 150, 150))
i = 0

for path in file_paths:
    img_name = file_paths[i]
    # read and prepare image
    x = imread('./Flag/{path}'.format(path=img_name))
    x = resize(x, (224, 224)) * 255
    x = np.expand_dims(x, 0)
   
    # use the model to classify whether it's a dog or cat
    prediction = model.predict(x) # this model will return 0 to 0.5 for dog, 0.5 to 1 for cat
    is_dog = bool(round(prediction[0][0])) # get the prediction numpy float to a boolean value

    # extract the coordinate based on image name
    coordinate = tuple(map(int, re.match(r"(\d+)_(\d+)", img_name).groups()))
    addpxl(flag_img, coordinate, is_dog)

    i = i + 1
    pprint("Done {done}/{remaining} (it was a {catdog})".format(done=i, remaining=len(file_paths), catdog=("dog" if is_dog else "cat")))

