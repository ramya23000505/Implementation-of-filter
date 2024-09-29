# Implementation-of-filter
### Date:
## Aim:
To implement filters for smoothing and sharpening the images in the spatial domain.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1
Import necessary libraries: OpenCV, NumPy, and Matplotlib.Read an image, convert it to RGB format, define an 11x11 averaging kernel, and apply 2D convolution filtering.Display the original and filtered images side by side using Matplotlib.

### Step2
Define a weighted averaging kernel (kernel2) and apply 2D convolution filtering to the RGB image (image2).Display the resulting filtered image (image4) titled 'Weighted Averaging Filtered' using Matplotlib's imshow function.

### Step3
Apply Gaussian blur with a kernel size of 11x11 and standard deviation of 0 to the RGB image (image2).Display the resulting Gaussian-blurred image (gaussian_blur) titled 'Gaussian Blurring Filtered' using Matplotlib's imshow function.

### Step4
Apply median blur with a kernel size of 11x11 to the RGB image (image2).Display the resulting median-blurred image (median) titled 'Median Blurring Filtered' using Matplotlib's imshow function.

### Step5
Define a Laplacian kernel (kernel3) and perform 2D convolution filtering on the RGB image (image2).Display the resulting filtered image (image5) titled 'Laplacian Kernel' using Matplotlib's imshow function.

### Step6
Apply the Laplacian operator to the RGB image (image2) using OpenCV's cv2.Laplacian function.Display the resulting image (new_image) titled 'Laplacian Operator' using Matplotlib's imshow function.

## Program:
### Developed By   : Ramya R
### Register Number: 21222320169
</br>

### 1. Smoothing Filters
#### Original Image
```
import cv2
import matplotlib.pyplot as plt
import numpy as np
image1 = cv2.imread("Ex_5_image.jpeg")
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")
```
#### i) Using Averaging Filter
```Python
kernel = np.ones((11, 11), np.float32) / 121
averaging_image = cv2.filter2D(image2, -1, kernel)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(averaging_image)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()
```
#### ii) Using Weighted Averaging Filter
```Python
kernel1 = np.array([[1, 2, 1],
                    [2, 4, 2],
                    [1, 2, 1]]) / 16

weighted_average_image = cv2.filter2D(image2, -1, kernel1)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(weighted_average_image)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()

```
#### iii) Using gaussian Filter
```Python
gaussian_blur = cv2.GaussianBlur(image2, (11, 11), 0)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()
```

#### iv) Using Median Filter
```Python
median_blur = cv2.medianBlur(image2, 11)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(median_blur)
plt.title("Median Filter")
plt.axis("off")
plt.show()
```

### 2. Sharpening Filters
#### i) Using Laplacian Linear Kernal
```Python
image1 = cv2.imread('Ex_5_image.jpeg')
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

kernel3 = np.array([[0,1,0], [1, -4,1],[0,1,0]])
image5 =cv2.filter2D(image2, -1, kernel3)
plt.imshow(image5)
plt.title('Laplacian Kernel')
```
#### ii) Using Laplacian Operator
```Python
laplacian = cv2.Laplacian(image2, cv2.CV_64F)
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 2)
plt.imshow(laplacian, cmap='gray')
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()
```

## OUTPUT:
#### Original Image
![Screenshot 2024-09-29 201312](https://github.com/user-attachments/assets/9961ab20-23e2-4f22-ba72-f3b260ba6bee)

### 1. Smoothing Filters
![Screenshot 2024-09-29 201427](https://github.com/user-attachments/assets/449a49e5-1ca7-44f9-930e-ad69663f295c)

#### i) Using Averaging Filter
![Screenshot 2024-09-29 201435](https://github.com/user-attachments/assets/23270573-0457-45bf-bfb7-f753dbac1f38)

#### ii) Using Gaussian Filter
![Screenshot 2024-09-29 201442](https://github.com/user-attachments/assets/b6a547c4-8287-4c0c-8e75-c601d7536b14)

#### iii) Using Median Filter
![Screenshot 2024-09-29 201450](https://github.com/user-attachments/assets/8c79be51-51b9-4c6b-b335-f7b66477adf4)

### 2. Sharpening Filters
#### i) Using Laplacian Kernal
![Screenshot 2024-09-29 201456](https://github.com/user-attachments/assets/cb4bf41a-af9f-43eb-9206-b9e0cda0ae39)

#### ii) Using Laplacian Operator
![Screenshot 2024-09-29 201501](https://github.com/user-attachments/assets/e2ec0650-551b-4edd-94e0-545db1de3aff)

## Result:
Thus the filters are designed for smoothing and sharpening the images in the spatial domain.
