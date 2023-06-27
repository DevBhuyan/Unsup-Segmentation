import torch
from skimg_local import rgb2hsv, hsv2rgb
import numpy as np
from utils import entropy_intrinsic, eps, hist_match
import cv2


def select_channels(img_RGB):

    img_RGB_norm = img_RGB / 255.0

    img_r_norm = img_RGB_norm[..., 0] / (img_RGB_norm[..., 0] + img_RGB_norm[..., 1] + img_RGB_norm[..., 2])

    img_v = np.max(img_RGB, axis=2)

    return (img_r_norm, img_v)


def calculate_GRAY(img_RGB):
    img = img_RGB.astype(np.float32) + eps

    print(img.shape)

    X = np.log(img.reshape(-1, 3))
    X_mean = np.mean(X, axis=0)
    X -= X_mean

    print(X.shape)

    U, S, V = np.linalg.svd(X.T, full_matrices=False)
    C = np.dot(X, U)
    C_reshaped = C.reshape(224, 224, -1)[:, :, 0]

    return C_reshaped



def calculate_Intrinsic_SA(img_RGB):
    """
    Returns the illumination invariant 'intrinsic' image and
    the shading attentuated representation for the skin lesion.

    Args:
        img_RGB (np.array): The RGB image of the skin lesion
    """
    img_torch = torch.from_numpy(img_RGB) + eps
    angle, projected = entropy_intrinsic(img_torch, calculate_intrinsic_img=True)
    projected_np = projected.cpu().detach().numpy()

    projected_norm = projected_np / 255.0
    img_HSV = rgb2hsv(img_RGB)
    matched = hist_match(img_HSV[..., 2], projected_norm)
    img_HSV[..., 2] = matched
    img_RGB_SA_norm = hsv2rgb(img_HSV)
    img_RGB_SA = img_RGB_SA_norm * 255

    return (projected_np, img_RGB_SA)