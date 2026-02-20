import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage import io, measure
from scipy import ndimage as ndi


TRAIN_PATH = "stage1_train"
NUM_IMAGES_TO_SHOW = 3



def load_image(sample_path):
    img_path = glob(os.path.join(sample_path, "images", "*"))[0]
    image = io.imread(img_path)

    if image.shape[-1] == 4: 
        image = image[:, :, :3]

    return image



def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Otsu threshold
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Ensure nuclei are white
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    return binary



def watershed_without_markers(image, binary):

   
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    local_max = distance > 0.3 * distance.max()

    markers = measure.label(local_max)

    labels = cv2.watershed(image.copy(), markers.astype(np.int32))

    return labels



def watershed_with_markers(image, binary):

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

   
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    _, sure_fg = cv2.threshold(distance, 0.5 * distance.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image.copy(), markers)

    return markers

def visualize(image, labels_no_markers, labels_markers):

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Without Markers")
    plt.imshow(labels_no_markers, cmap="nipy_spectral")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Marker-Controlled")
    plt.imshow(labels_markers, cmap="nipy_spectral")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def run():

    sample_folders = os.listdir(TRAIN_PATH)

    for sample in sample_folders[:NUM_IMAGES_TO_SHOW]:

        sample_path = os.path.join(TRAIN_PATH, sample)

        image = load_image(sample_path)
        binary = preprocess(image)

        labels_no_markers = watershed_without_markers(image, binary)
        labels_markers = watershed_with_markers(image, binary)

        print(f"\nSample: {sample}")
        print("Objects (without markers):", len(np.unique(labels_no_markers)) - 1)
        print("Objects (with markers):", len(np.unique(labels_markers)) - 1)

        visualize(image, labels_no_markers, labels_markers)


if __name__ == "__main__":
    run()