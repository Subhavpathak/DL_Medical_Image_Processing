import os
from glob import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import threshold_sauvola
from skimage.morphology import remove_small_objects
from scipy import ndimage

# ---------- Metrics ----------

def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return 2 * intersection / total

def jaccard_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union

# ---------- Preprocessing ----------

def preprocess(img):
    clahe = cv2.createCLAHE(2.0, (8,8))
    img = clahe.apply(img)
    return img

# ---------- Threshold Methods ----------

def otsu_threshold(img):
    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def sauvola_threshold(img, window=25, k=0.2):
    thresh = threshold_sauvola(img, window_size=window, k=k)
    binary = img > thresh
    return (binary.astype(np.uint8) * 255)

# ---------- Postprocessing ----------

def clean_mask(mask, min_size=200):
    mask = mask > 127
    mask = remove_small_objects(mask, min_size=min_size)
    mask = ndimage.binary_fill_holes(mask)
    return (mask.astype(np.uint8) * 255)

# ---------- Main ----------

def main(images_path, masks_path):

    image_files = sorted(glob(os.path.join(images_path, "*.png")))
    results = []

    for img_path in tqdm(image_files):
        name = os.path.basename(img_path)
        mask_path = os.path.join(masks_path, name)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = gt > 127

        img = preprocess(img)

        # Otsu
        otsu = clean_mask(otsu_threshold(img))
        dice_o = dice_score(otsu > 127, gt)
        jac_o = jaccard_score(otsu > 127, gt)

        # Sauvola
        sau = clean_mask(sauvola_threshold(img))
        dice_s = dice_score(sau > 127, gt)
        jac_s = jaccard_score(sau > 127, gt)

        results.append([dice_o, jac_o, dice_s, jac_s])

    df = pd.DataFrame(results, columns=[
        "Otsu_Dice", "Otsu_Jaccard",
        "Sauvola_Dice", "Sauvola_Jaccard"
    ])

    print("\n===== FINAL RESULTS =====")
    print(df.mean())

    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main("data/images", "data/masks")
