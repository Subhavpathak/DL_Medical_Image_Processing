# Brain MRI Tumor Segmentation: Otsu vs Sauvola

## Objective
Segment tumor regions in MRI slices using:
- Global Otsu Thresholding
- Adaptive Sauvola Thresholding

## Evaluation Metrics
- Dice Score
- Jaccard Index

## Observations
Otsu uses a single global threshold for the entire image, while Sauvola computes a local adaptive threshold based on neighborhood statistics.

MRI images often contain non-uniform intensity distribution. Because of this, Sauvola generally performs better than Otsu.

However, both methods struggle with complex tumor boundaries and intensity overlap between tumor and healthy tissue. These classical thresholding approaches lack robustness compared to deep learning-based segmentation models.
