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

WINDOW_SIZE = 51
K_NIBLACK = 0.2
K_SAUVOLA = 0.3  
MIN_OBJECT_SIZE = 50



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
    if img is None:
        print("Failed to load image:", img_path)
        return None

    green = img[:, :, 1]
    green = cv2.equalizeHist(green)
    green = cv2.GaussianBlur(green, (5,5), 0)

    return green


def load_mask(mask_path):
    try:
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        return mask > 0
    except:
        print("Failed to load mask:", mask_path)
        return None


def apply_niblack(image):
    thresh = threshold_niblack(image, window_size=WINDOW_SIZE, k=K_NIBLACK)
    binary = image > thresh
    binary = remove_small_objects(binary, MIN_OBJECT_SIZE)
    return binary


def apply_sauvola(image):
    thresh = threshold_sauvola(image, window_size=WINDOW_SIZE, k=K_SAUVOLA)
    binary = image > thresh
    binary = remove_small_objects(binary, MIN_OBJECT_SIZE)
    return binary


def evaluate(split="test"):

    if split == "training":
        img_dir = os.path.join(BASE_PATH, "training", "images")
        gt_dir = os.path.join(BASE_PATH, "training", "1st_manual")
    else:
        img_dir = os.path.join(BASE_PATH, "test", "images")
        gt_dir = os.path.join(BASE_PATH, "test", "mask")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))
    gt_paths = sorted(glob(os.path.join(gt_dir, "*")))

    print(f"\nProcessing {split.upper()} set")
    print("Images found:", len(img_paths))
    print("Masks found:", len(gt_paths))

    results = []

    for img_path, gt_path in tqdm(list(zip(img_paths, gt_paths))):

        image = preprocess(img_path)
        gt = load_mask(gt_path)

        if image is None or gt is None:
            continue

        pred_n = apply_niblack(image)
        dice_n, jac_n, sens_n = compute_metrics(pred_n, gt)

        pred_s = apply_sauvola(image)
        dice_s, jac_s, sens_s = compute_metrics(pred_s, gt)

        results.append({
            "Image": os.path.basename(img_path),
            "Niblack_Dice": dice_n,
            "Niblack_Jaccard": jac_n,
            "Niblack_Sensitivity": sens_n,
            "Sauvola_Dice": dice_s,
            "Sauvola_Jaccard": jac_s,
            "Sauvola_Sensitivity": sens_s
        })

    return pd.DataFrame(results)



def show_visual_results(split="test", num_images=3):

    if split == "training":
        img_dir = os.path.join(BASE_PATH, "training", "images")
        gt_dir = os.path.join(BASE_PATH, "training", "1st_manual")
    else:
        img_dir = os.path.join(BASE_PATH, "test", "images")
        gt_dir = os.path.join(BASE_PATH, "test", "mask")

    img_paths = sorted(glob(os.path.join(img_dir, "*.tif")))[:num_images]
    gt_paths = sorted(glob(os.path.join(gt_dir, "*")))[:num_images]

    for img_path, gt_path in zip(img_paths, gt_paths):

        image = preprocess(img_path)
        gt = load_mask(gt_path)

        pred_n = apply_niblack(image)
        pred_s = apply_sauvola(image)

        plt.figure(figsize=(12,6))

        plt.subplot(1,4,1)
        plt.title("Original")
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

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":

    df_train = evaluate("training")
    df_test = evaluate("test")

    print("\n==============================")
    print("TRAIN AVERAGE RESULTS")
    print("==============================")
    print(df_train.mean(numeric_only=True))

    print("\n==============================")
    print("TEST AVERAGE RESULTS")
    print("==============================")
    print(df_test.mean(numeric_only=True))

    df_train.to_csv("train_results.csv", index=False)
    df_test.to_csv("test_results.csv", index=False)

    print("\nResults saved.")

    print("\nShowing visual results from TEST set...")
    show_visual_results("test", num_images=3)
