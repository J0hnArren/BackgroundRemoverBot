from PIL import Image as Img
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pandas as pd
from u_2_net.data_loader import RescaleT
from u_2_net.data_loader import ToTensorLab
from u_2_net.data_loader import SalObjDataset
from u_2_net.model.u2net import U2NETP


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def bilateral_filter_convert(origin_input):
    imidx = origin_input['imidx']
    label = origin_input['label']
    img = np.asarray(origin_input['image'])
    img = cv.bilateralFilter(img, 5, 75, 75)

    return {'imidx': imidx, 'image': img, 'label': label}


def equalize_hist_convert(origin_input):
    imidx = origin_input['imidx']
    label = origin_input['label']
    img = np.asarray(origin_input['image'])
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    img[:, :, -1] = cv.equalizeHist(img[:, :, -1])
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    return {'imidx': imidx, 'image': img, 'label': label}


def clashe_convert(origin_input):
    imidx = origin_input['imidx']
    label = origin_input['label']
    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(24, 32))
    img = np.asarray(origin_input['image'])
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img[:, :, -1] = clahe.apply(img[:, :, -1])
    img = cv.cvtColor(img, cv.COLOR_HSV2BGR)

    return {'imidx': imidx, 'image': img, 'label': label}


def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


class Model_Pytorch:
    def __init__(self):
        self.model_dir = "models/u2net_model.pth"

        self.net = U2NETP(3, 1)

        # if torch.cuda.is_available():
        #     self.net.load_state_dict(torch.load(self.model_dir))
        #     self.net.cuda()
        # else:
        #     self.net.load_state_dict(torch.load(self.model_dir, map_location=torch.device("cpu")))

        self.net.load_state_dict(torch.load(self.model_dir, map_location=torch.device("cpu")))

        self.net.eval()

        self.root_out = "./data/"
        self.THRESHOLD = 0.9
        self.VALID_THRESHOLD = 0.2
        self.BATCH_SIZE = 2
        # self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.DEVICE = "cpu"

    def pred_unet_clahe(self, paths):
        salobj_dataset = SalObjDataset(img_name_list=paths, lbl_name_list=[],
                                       transform=transforms.Compose(
                                           [clashe_convert, RescaleT(320), ToTensorLab(flag=0)]))
        salobj_dataloader = DataLoader(salobj_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0)

        self.net.to(self.DEVICE)

        error_log = []
        for i, batch in enumerate(salobj_dataloader):
            self.net.eval()
            img_batch = batch['image']  # (batch size, channel, row, columns)
            images = img_batch.type(torch.float32)  # change torch.double -> torch.float
            images = images.to(self.DEVICE)
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = self.net(images)

            predict = d5[:, 0, :, :]  # (batch size, 1, row, columns)
            predict = norm_pred(predict)

            del d1, d2, d3, d4, d5, d6, d7

            predict = predict.squeeze()  # (batch size, row, columns)
            predict_np = predict.cpu().numpy()

            # Masked image - using threshold you can soften/sharpen mask boundaries
            predict_np[predict_np > self.THRESHOLD] = 1
            predict_np[predict_np <= self.THRESHOLD] = 0

            for j in range(len(predict_np)):
                file = paths[i * self.BATCH_SIZE + j]
                mask_np = predict_np[j]
                mask = Img.fromarray(mask_np * 255).convert('RGB')
                image = Img.open(paths[i * self.BATCH_SIZE + j])

                mask = mask.resize((image.width, image.height), resample=Img.BILINEAR)
                back = Img.new("RGB", (image.width, image.height), (255, 255, 255))  # WHITE Backgroud
                mask = mask.convert('L')
                im_out = Img.composite(image, back, mask)

                mask_rs = np.array(mask)

                x, y, w, h = cv.boundingRect(mask_rs)

                # used to sift invalid image by the "1"(white) area / rectangle area(white and black)
                if (x, y, w, h) == (0, 0, 0, 0) or (((mask_rs != 0).sum()) / (w * h)) < self.VALID_THRESHOLD:
                    crop_img = np.array(image)
                    error_log.append(file)
                    print('Failed:\t', file)
                else:
                    im_out_np = np.array(im_out)
                    crop_img = im_out_np  # [ymin:ymax, xmin:xmax]

                cv.imwrite(file, crop_img)

                del image


    def cut_image_bg_pytorch(self, img_path):
        paths = ["./data/sample.jpg", img_path]
        self.pred_unet_clahe(paths)
        self.pred_unet_clahe(paths)
