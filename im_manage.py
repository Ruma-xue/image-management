# coding=utf-8
import os
from PIL import Image


def im_management(filename, lable):
    cat = './im_test'
    cat_im = './im_test/cat_image'
    dog_im = './im_test/dog_image'
    im = os.path.join(cat, filename)
    file = Image.open(im)
    if lable is 'Cat':
        file.save(cat_im + '/' + filename)
        os.remove(im)

    elif lable is 'Dog':
        file.save(dog_im + '/' + filename)
        os.remove(im)
