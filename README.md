
# Image Segmentation Studio

An interactive image processing application that demonstrates five classical image segmentation techniques using Python, OpenCV, NumPy, and Streamlit.

## Project Overview

Image segmentation is an important image processing technique used to divide an image into meaningful regions based on characteristics such as pixel intensity and similarity.

This project provides an interactive platform to implement and explore five classical image segmentation techniques. Users can upload images, select an algorithm, adjust its parameters, view the segmentation results, and download the processed images.

## Implemented Techniques

### 1. Thresholding
Separates an image into foreground and background based on a manually selected intensity threshold.

### 2. Otsu Thresholding
Automatically determines an optimal threshold value based on the image intensity histogram.

### 3. Region Growing
Starts from a selected seed pixel and expands the region by including neighbouring pixels with similar intensity values.

### 4. Region Splitting
Recursively divides an image into smaller regions when the intensity variation within a region is too high.

### 5. Region Merging
Combines adjacent regions that have similar average pixel intensities.

## Features

- Upload images in JPG, JPEG, and PNG formats.
- Interactive selection of segmentation techniques.
- Adjustable algorithm parameters.
- Original and segmented image comparison.
- Download processed images.
- Simple and interactive web interface.

## Technologies Used

- Python
- OpenCV
- NumPy
- Pillow
- Streamlit

## Project Structure

```text
IPA-Image-Studio/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd IPA-Image-Studio
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Upload an image.
2. Select a segmentation technique from the sidebar.
3. Adjust the available parameters.
4. View the grayscale and segmented images.
5. Download the processed result.

## Project Team

This project is developed collaboratively by a team of three students as part of the Image Processing and Applications course.

- Member 1: AESHNA SACHEE
- Member 2: ABHIJEET ORAON
- Member 3: ABHINAV DAKSH 

## Academic Project

Course: Image Processing and Applications (IPA)

Project Title: Image Segmentation Using Classical Image Processing Techniques
