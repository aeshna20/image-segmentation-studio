
import streamlit as st
import cv2
import numpy as np
from PIL import Image


# ==========================================
# REGION SPLITTING FUNCTION
# ==========================================

def region_splitting(image, std_threshold, min_size):

    height, width = image.shape
    result = np.zeros_like(image)
    boundaries = np.zeros_like(image)

    def split_region(x, y, w, h):

        region = image[y:y+h, x:x+w]

        if region.size == 0:
            return

        variation = np.std(region)

        if (
            variation <= std_threshold
            or (w <= min_size and h <= min_size)
            or w < 2
            or h < 2
        ):
            result[y:y+h, x:x+w] = int(np.mean(region))
            return

        half_w = w // 2
        half_h = h // 2

        split_region(x, y, half_w, half_h)
        split_region(x + half_w, y, w - half_w, half_h)
        split_region(x, y + half_h, half_w, h - half_h)
        split_region(
            x + half_w,
            y + half_h,
            w - half_w,
            h - half_h
        )

        boundaries[y, x:x+w] = 255
        boundaries[y+h-1, x:x+w] = 255
        boundaries[y:y+h, x] = 255
        boundaries[y:y+h, x+w-1] = 255

    split_region(0, 0, width, height)

    return result, boundaries


# ==========================================
# REGION MERGING FUNCTION
# ==========================================

def region_merging(image, merge_threshold, cell_size):

    height, width = image.shape

    # Divide image into initial small regions
    rows = (height + cell_size - 1) // cell_size
    cols = (width + cell_size - 1) // cell_size

    means = []
    counts = []
    grid = np.zeros((rows, cols), dtype=np.int32)

    region_id = 0

    for r in range(rows):
        for c in range(cols):

            y1 = r * cell_size
            y2 = min(y1 + cell_size, height)

            x1 = c * cell_size
            x2 = min(x1 + cell_size, width)

            region = image[y1:y2, x1:x2]

            means.append(float(np.mean(region)))
            counts.append(region.size)

            grid[r, c] = region_id
            region_id += 1

    # Union-Find data structure
    parent = list(range(region_id))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    # Find adjacent initial regions
    edges = set()

    for r in range(rows):
        for c in range(cols):

            current = int(grid[r, c])

            if c + 1 < cols:
                right = int(grid[r, c + 1])
                edges.add((min(current, right), max(current, right)))

            if r + 1 < rows:
                bottom = int(grid[r + 1, c])
                edges.add((min(current, bottom), max(current, bottom)))

    # Process similar neighbouring regions first
    edge_list = sorted(
        [
            (
                abs(means[a] - means[b]),
                a,
                b
            )
            for a, b in edges
        ]
    )

    # Merge regions with similar average intensity
    for difference, a, b in edge_list:

        root_a = find(a)
        root_b = find(b)

        if root_a == root_b:
            continue

        mean_difference = abs(
            means[root_a] - means[root_b]
        )

        if mean_difference <= merge_threshold:

            total_count = counts[root_a] + counts[root_b]

            combined_mean = (
                means[root_a] * counts[root_a]
                + means[root_b] * counts[root_b]
            ) / total_count

            parent[root_b] = root_a

            means[root_a] = combined_mean
            counts[root_a] = total_count

    # Assign final merged region labels
    final_grid = np.zeros_like(grid)

    for r in range(rows):
        for c in range(cols):
            final_grid[r, c] = find(int(grid[r, c]))

    # Create output image
    result = np.zeros_like(image)

    for r in range(rows):
        for c in range(cols):

            y1 = r * cell_size
            y2 = min(y1 + cell_size, height)

            x1 = c * cell_size
            x2 = min(x1 + cell_size, width)

            root = int(final_grid[r, c])

            result[y1:y2, x1:x2] = int(
                np.clip(means[root], 0, 255)
            )

    return result, len(np.unique(final_grid))


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Image Segmentation Studio",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Image Segmentation Studio")

st.write(
    "An interactive application demonstrating five classical "
    "image segmentation techniques using Python and OpenCV."
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("Segmentation Methods")

method = st.sidebar.selectbox(
    "Choose a technique",
    [
        "Thresholding",
        "Otsu Thresholding",
        "Region Growing",
        "Region Splitting",
        "Region Merging"
    ]
)


# ==========================================
# DESCRIPTIONS
# ==========================================

descriptions = {
    "Thresholding":
        "Separates foreground and background using a manually selected intensity threshold.",

    "Otsu Thresholding":
        "Automatically calculates an optimal threshold from the image histogram.",

    "Region Growing":
        "Expands a region from a seed pixel by including neighbouring pixels with similar intensity.",

    "Region Splitting":
        "Recursively divides an image into smaller regions when intensity variation is high.",

    "Region Merging":
        "Combines adjacent regions with similar average pixel intensities."
}

st.subheader(f"Selected Method: {method}")
st.write(descriptions[method])


# ==========================================
# IMAGE UPLOAD
# ==========================================

st.divider()

st.subheader("How to Use the Application")

st.markdown("""
1. Upload an image in JPG, JPEG, or PNG format.
2. Select one of the five segmentation techniques from the sidebar.
3. Adjust the algorithm parameters where available.
4. Observe the grayscale and segmented images.
5. Download the processed image using the download button.

**Tip:** Experiment with different images and parameter values
to understand how each segmentation technique works.
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    gray_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    height, width = gray_image.shape

    st.subheader("Original Image")

    st.image(
    image_array,
    caption="Uploaded Original Image",
    width=450)

    # ======================================
    # 1. THRESHOLDING
    # ======================================

    if method == "Thresholding":

        st.subheader("Manual Thresholding")

        threshold_value = st.slider(
            "Select Threshold Value",
            0, 255, 127
        )

        _, segmented_image = cv2.threshold(
            gray_image,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        st.write(f"Selected Threshold: {threshold_value}")


    # ======================================
    # 2. OTSU THRESHOLDING
    # ======================================

    elif method == "Otsu Thresholding":

        st.subheader("Automatic Otsu Thresholding")

        otsu_threshold, segmented_image = cv2.threshold(
            gray_image,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        st.write(
            f"Automatically Calculated Threshold: "
            f"{otsu_threshold:.2f}"
        )


    # ======================================
    # 3. REGION GROWING
    # ======================================

    elif method == "Region Growing":

        st.subheader("Region Growing Segmentation")

        col1, col2 = st.columns(2)

        with col1:
            seed_x = st.slider(
                "Seed X Coordinate",
                0,
                width - 1,
                width // 2
            )

        with col2:
            seed_y = st.slider(
                "Seed Y Coordinate",
                0,
                height - 1,
                height // 2
            )

        tolerance = st.slider(
            "Intensity Similarity Threshold",
            1, 100, 20
        )

        seed_intensity = int(gray_image[seed_y, seed_x])

        mask = np.zeros(
            (height + 2, width + 2),
            dtype=np.uint8
        )

        flags = (
            4
            | cv2.FLOODFILL_MASK_ONLY
            | cv2.FLOODFILL_FIXED_RANGE
            | (255 << 8)
        )

        cv2.floodFill(
            gray_image.copy(),
            mask,
            (seed_x, seed_y),
            0,
            loDiff=tolerance,
            upDiff=tolerance,
            flags=flags
        )

        segmented_image = mask[1:-1, 1:-1].copy()

        st.write(f"Seed Pixel Intensity: {seed_intensity}")

        marked_image = image_array.copy()

        cv2.circle(
            marked_image,
            (seed_x, seed_y),
            8,
            (255, 0, 0),
            -1
        )

        st.image(
            marked_image,
            caption="Selected Seed Pixel",
            use_container_width=True
        )


    # ======================================
    # 4. REGION SPLITTING
    # ======================================

    elif method == "Region Splitting":

        st.subheader("Quadtree Region Splitting")

        std_threshold = st.slider(
            "Homogeneity Threshold",
            1, 100, 20
        )

        min_size = st.select_slider(
            "Minimum Region Size",
            options=[4, 8, 16, 32, 64],
            value=16
        )

        segmented_image, boundaries = region_splitting(
            gray_image,
            std_threshold,
            min_size
        )

        boundary_image = gray_image.copy()
        boundary_image[boundaries == 255] = 255

        st.image(
            boundary_image,
            caption="Region Splitting Boundaries",
            use_container_width=True
        )


    # ======================================
    # 5. REGION MERGING
    # ======================================

    elif method == "Region Merging":

        st.subheader("Region Merging Segmentation")

        merge_threshold = st.slider(
            "Intensity Similarity Threshold",
            1, 100, 20
        )

        cell_size = st.select_slider(
            "Initial Region Size (pixels)",
            options=[8, 16, 32, 64],
            value=32
        )

        segmented_image, region_count = region_merging(
            gray_image,
            merge_threshold,
            cell_size
        )

        st.write(
            f"Number of regions after merging: {region_count}"
        )

        st.write(
            "Neighbouring regions with similar average "
            "intensities have been combined."
        )


    # ======================================
    # DISPLAY RESULTS
    # ======================================

    st.divider()
    st.subheader("Segmentation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            gray_image,
            caption="Grayscale Image",
            use_container_width=True
        )

    with col2:
        st.image(
            segmented_image,
            caption="Segmented Image",
            use_container_width=True
        )

    # Download result
    success, encoded_image = cv2.imencode(
        ".png",
        segmented_image
    )

    if success:
        st.download_button(
            label="Download Segmented Image",
            data=encoded_image.tobytes(),
            file_name="segmented_image.png",
            mime="image/png"
        )

else:

    st.info("Please upload an image to begin.")
