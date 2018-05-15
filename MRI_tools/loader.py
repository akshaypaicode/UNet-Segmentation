# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:39:53 2018

@author: akshay
"""

import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-6
from scipy.io import loadmat
import numpy as np


def load_nii(path):
    image_obj = nib.load(path)
    image = image_obj.get_data()

    return image_obj, image


def load_mat(path):
    image_obj = loadmat(path)
    name = [n for n in image_obj if isinstance(image_obj[n], np.ndarray)]
    assert len(name) == 1
    image = image_obj[name[0]]

    return image_obj, image
