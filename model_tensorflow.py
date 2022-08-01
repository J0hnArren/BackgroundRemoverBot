from skimage import io
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image


def transfer_mask(src, img_path, orig_path):
    img = Image.open(img_path)
    orig = Image.open(orig_path)

    img = img.convert("RGBA")
    orig = orig.convert("RGBA")
    datas = img.getdata()
    orig_data = orig.getdata()

    new_data = []
    colors = [247, 231, 230]
    for item, item_orig in zip(datas, orig_data):
        if item[0] in colors or item[1] in colors or item[2] in colors:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item_orig)

    img.putdata(new_data)
    img.save(f"{src}w_bg.png", "PNG")


class Model_Tensorflow:

    def __init__(self):
        self.img_size = (256, 256)
        self.model = keras.models.load_model("models/model_epoch_4_val_loss_0.209310.h5")

    def cut_image_bg_tf(self, src, filename):
        path = src + filename

        im = io.imread(path)
        im_orig_size = im.shape[:-1]
        im = cv2.resize(im, self.img_size)
        im = np.array(im) / 255

        im = im.reshape((1,) + im.shape)

        pred = self.model.predict(im)

        p = pred.copy()
        p = p.reshape(p.shape[1:-1])

        p[np.where(p > .2)] = 1
        p[np.where(p < .2)] = 0

        im = io.imread(path)
        im = cv2.resize(im, self.img_size)
        im = np.array(im)

        im[:, :, 0] = im[:, :, 0] * p
        im[:, :, 0][np.where(p != 1)] = 247
        im[:, :, 1] = im[:, :, 1] * p
        im[:, :, 1][np.where(p != 1)] = 231
        im[:, :, 2] = im[:, :, 2] * p
        im[:, :, 2][np.where(p != 1)] = 230

        im = cv2.resize(im, im_orig_size[::-1])
        im = Image.fromarray(im)
        im.save(f"{src}without_bg_256_{filename}")

        transfer_mask(src, f"{src}without_bg_256_{filename}", path)
