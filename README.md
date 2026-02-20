# DL Medical Image Processing

This repository contains three classical medical image segmentation assignments implemented using Python, OpenCV, and Scikit-Image.

The focus is on:
- Global vs adaptive thresholding
- Vessel extraction
- Marker-controlled watershed
- Over-segmentation control
- Quantitative evaluation using Dice, Jaccard, and Sensitivity


------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------

Install dependencies using:

pip install -r requirements.txt

Main libraries used:
- numpy
- opencv-python
- matplotlib
- pandas
- scikit-image
- scipy
- pillow
- tqdm


------------------------------------------------------------
ASSIGNMENT 1 – Brain MRI Segmentation (Otsu vs Sauvola)
------------------------------------------------------------

Task:
Segment brain MRI images using global and adaptive thresholding.

Methods:
- CLAHE preprocessing
- Otsu Thresholding (global)
- Sauvola Thresholding (adaptive)
- Morphological cleaning
- Dice Score
- Jaccard Index

Dataset:
Brain MRI segmentation dataset :
[https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

Dataset Structure:

data/
 ├── images/
 └── masks/

Run:
python assignment1_Brain_MRI.py

Output:
- Prints average Dice and Jaccard scores
- Saves per-image results in results.csv

Learning:
Adaptive thresholding (Sauvola) performs better in non-uniform illumination compared to global Otsu thresholding.


------------------------------------------------------------
ASSIGNMENT 2 – Retinal Vessel Extraction (Niblack vs Sauvola)
------------------------------------------------------------

Task:
Extract retinal blood vessels from fundus images.

Methods:
- Green channel extraction
- Histogram equalization
- Gaussian smoothing
- Niblack thresholding
- Sauvola thresholding
- Small object removal
- Dice, Jaccard, Sensitivity evaluation

Dataset:
DRIVE Retinal Vessel Dataset:
[https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/data)

Expected Structure:

DRIVE/
 ├── training/
 │     ├── images/
 │     └── 1st_manual/
 └── test/
       ├── images/
       └── mask/

Run:
python assignment3_Retinal_Vessel.py

Output:
- Prints average metrics for training and test sets
- Saves:
  - train_results.csv
  - test_results.csv
- Displays segmentation visualization

Learning:
Local thresholding improves thin vessel detection.
Parameter tuning affects sensitivity vs over-segmentation trade-off.


------------------------------------------------------------
ASSIGNMENT 3 – Cell Nuclei Separation (Watershed)
------------------------------------------------------------

Task:
Separate touching nuclei using watershed segmentation.

Methods:
- Otsu thresholding
- Distance transform
- Watershed without markers
- Marker-controlled watershed
- Morphological operations

Dataset:
Data Science Bowl 2018 – Cell Nuclei:
[https://www.kaggle.com/competitions/data-science-bowl-2018/data](https://www.kaggle.com/competitions/data-science-bowl-2018/data)

Download:
stage1_train.zip

Expected Structure:

stage1_train/
 ├── sample_id_1/
 │     ├── images/
 │     └── masks/
 ├── sample_id_2/
 └── ...

Each sample contains:
- One image
- Multiple nucleus masks

Run:
python assignment4_Cell_Nuclei_Sep.py

Output:
- Displays segmentation comparison
- Prints object counts:
  - Without markers
  - With marker control

Learning:
Watershed without markers causes over-segmentation.
Marker-controlled watershed reduces false splits and improves object separation.


------------------------------------------------------------
SUMMARY OF TECHNIQUES
------------------------------------------------------------

Assignment 1:
Global vs adaptive thresholding for medical image segmentation.

Assignment 2:
Local thresholding methods for thin structure extraction.

Assignment 3:
Watershed segmentation with over-segmentation control.


------------------------------------------------------------
AUTHOR
------------------------------------------------------------

Subhav Kumar  
DL Medical Image Processing
