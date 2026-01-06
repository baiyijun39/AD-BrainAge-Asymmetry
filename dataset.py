import os
import copy
import numpy as np
import pandas as pd
import torch
import nibabel as nib
import SimpleITK as sitk
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sfcn_torch import SFCN
from torch.cuda.amp import GradScaler, autocast
from monai.transforms import ToTensor, RandFlip, RandAffine, RandBiasField, NormalizeIntensity, Compose

TABLE_TRAIN_PATH = "/raid/byj_file/T1_and_FLAIR_new/brain_age_data/data/byj/dataset_label/new_data_train.csv"
TABLE_TEST_PATH = "/raid/byj_file/T1_and_FLAIR_new/brain_age_data/data/byj/dataset_label/new_data_test.csv"
data_path = "/raid/byj_file/T1_and_FLAIR_new/brain_age_data/data/byj/"

table_train = pd.read_csv(TABLE_TRAIN_PATH)
table_test = pd.read_csv(TABLE_TEST_PATH)

train_age = dict(zip(table_train["id"], table_train["Age"]))
test_age = dict(zip(table_test["id"], table_test["Age"]))

train_area = dict(zip(table_train["id"], table_train["area"]))
test_area = dict(zip(table_test["id"], table_test["area"]))


def dataset(data_path):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    area_train = []
    area_test = []

    train_ids = set(train_age.keys())
    test_ids = set(test_age.keys())

    for root, dirs, files in os.walk(data_path):
        for d in dirs:
            if d in train_ids:
                image_path = os.path.join(root, d, "smwp1T1_resample.nii")
                print(image_path)
                if os.path.exists(image_path):
                    image = sitk.ReadImage(image_path)
                    image_array = sitk.GetArrayFromImage(image).reshape(1, 121, 145, 121)
                    x_train.append(image_array)
                    y_train.append(train_age[d])
                    area_train.append(train_area[d])
            elif d in test_ids:
                image_path = os.path.join(root, d, "smwp1T1_resample.nii")
                if os.path.exists(image_path):
                    image = sitk.ReadImage(image_path)
                    print(image_path)
                    image_array = sitk.GetArrayFromImage(image).reshape(1, 121, 145, 121)
                    x_test.append(image_array)
                    y_test.append(test_age[d])
                    area_test.append(test_area[d])

    print(len(x_train))
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))

    return x_train, y_train, x_test, y_test
