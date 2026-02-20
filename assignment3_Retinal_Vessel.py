import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_niblack, threshold_sauvola
from skimage.morphology import remove_small_objects
from PIL import Image
import matplotlib.pyplot as plt



BASE_PATH = "DRIVE"
WINDOW_SIZE = 25
K_NIBLACK = -0.2
K_SAUVOLA = 0.2
MIN_OBJECT_SIZE = 20
NUM_VISUALIZE = 3



def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()

    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    jaccard = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)

    return dice, jaccard, sensitivity


def preprocess(img_path):
    img = cv2.imread(img_path)
    green = img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    green = clahe.apply(green)

    green = cv2.GaussianBlur(green, (3,3), 0)

    return green


def load_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    return mask > 0


def load_fov(image_name):
    fov_path = os.path.join(BASE_PATH, "training", "mask",
                            image_name.replace(".tif", "_mask.gif"))
    fov = Image.open(fov_path).convert("L")
    fov = np.array(fov)
    return fov > 0


def apply_niblack(image):
    thresh = threshold_niblack(image, window_size=WINDOW_SIZE, k=K_NIBLACK)
    binary = image < thresh
    binary = remove_small_objects(binary, MIN_OBJECT_SIZE)
    return binary


def apply_sauvola(image):
    thresh = threshold_sauvola(image, window_size=WINDOW_SIZE, k=K_SAUVOLA)
    binary = image < thresh
    binary = remove_small_objects(binary, MIN_OBJECT_SIZE)
    return binary


def evaluate_training():

    img_dir = os.path.join(BASE_PATH, "training", "images")
    gt_dir = os.path.join(BASE_PATH, "training", "1st_manual")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*")))

    results = []

    for i, (img_path, gt_path) in enumerate(tqdm(list(zip(img_paths, gt_paths)))):

        image_name = os.path.basename(img_path)

        image = preprocess(img_path)
        gt = load_mask(gt_path)
        fov = load_fov(image_name)

        gt = gt & fov

        pred_n = apply_niblack(image) & fov
        pred_s = apply_sauvola(image) & fov

        dice_n, jac_n, sens_n = compute_metrics(pred_n, gt)
        dice_s, jac_s, sens_s = compute_metrics(pred_s, gt)

        results.append({
            "Image": image_name,
            "Niblack_Dice": dice_n,
            "Niblack_Jaccard": jac_n,
            "Niblack_Sensitivity": sens_n,
            "Sauvola_Dice": dice_s,
            "Sauvola_Jaccard": jac_s,
            "Sauvola_Sensitivity": sens_s
        })

        if i < NUM_VISUALIZE:
            plt.figure(figsize=(15,5))

            plt.subplot(1,4,1)
            plt.title("Preprocessed")
            plt.imshow(image, cmap="gray")
            plt.axis("off")

            plt.subplot(1,4,2)
            plt.title("Ground Truth")
            plt.imshow(gt, cmap="gray")
            plt.axis("off")

            plt.subplot(1,4,3)
            plt.title("Niblack")
            plt.imshow(pred_n, cmap="gray")
            plt.axis("off")

            plt.subplot(1,4,4)
            plt.title("Sauvola")
            plt.imshow(pred_s, cmap="gray")
            plt.axis("off")

            plt.suptitle(image_name)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)



if __name__ == "__main__":

    df_train = evaluate_training()

    print("\n===== RESULTS =====")
    print(df_train.mean(numeric_only=True))

    df_train.to_csv("train_results.csv", index=False)
