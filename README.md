# Stitching-Images-using-OpenCV-and-RANSAC-algorithm

STEPS:
1. Reading images from a folder called 'data'.
2. Finding Key points in the image using SIFT (Scale-Invariant-Feature-Transform) to detect and describe features in an image.
3. Matching the key points in order to detect matching features between the images by computing euclidean distance.
4. Once matching features are obtained computing squared euclidean distance between images.
5. Stitching images together to estimate a homography matrix using our matched features.
6. Applying a warping transformation using the homography matrix.
