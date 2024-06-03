import os
import shutil
import numpy as np
import cv2 as cv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define directories
train_directory = 'input/lab3/train'
template_directory = 'input/lab3'
output_directory = 'lab3-result'


def clear_output_directory(directory):
    """Clear the output directory before processing."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


# Clear the output directory
clear_output_directory(output_directory)


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


def pad_images_to_same_height(img1, img2):
    """Pad images to have the same height"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 < h2:
        img1 = cv.copyMakeBorder(img1, 0, h2 - h1, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    elif h2 < h1:
        img2 = cv.copyMakeBorder(img2, 0, h1 - h2, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return img1, img2


def draw_thick_matches(img1, kp1, img2, kp2, matches, color=(0, 255, 0), thickness=2):
    """Draw matches with thick lines."""
    img1, img2 = pad_images_to_same_height(img1, img2)

    # Create a new output image that concatenates the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype='uint8')
    output_img[:h1, :w1] = img1
    output_img[:h2, w1:] = img2

    # Draw the matches
    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv.circle(output_img, (int(x1), int(y1)), 5, color, thickness)
        cv.circle(output_img, (int(x2) + w1, int(y2)), 5, color, thickness)
        cv.line(output_img, (int(x1), int(y1)), (int(x2) + w1, int(y2)), color, thickness)

    return output_img


def process_and_save(method_name, detector, matcher, train_directory, template_directory, output_directory,
                     use_knn=False, ratio_test=False):
    method_dir = os.path.join(output_directory, method_name)
    os.makedirs(method_dir, exist_ok=True)
    logging.info(f"Processing with method: {method_name}")

    train_files = [os.path.join(train_directory, f) for f in os.listdir(train_directory) if is_image_file(f)]
    template_files = [os.path.join(template_directory, f) for f in os.listdir(template_directory) if is_image_file(f)]

    for train_file in train_files:
        original_img = cv.imread(train_file)
        if original_img is None:
            logging.warning(f"Failed to load training image {train_file}")
            continue

        img2_gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
        img2_color = original_img.copy()  # Create a copy for drawing
        mask = np.ones(img2_gray.shape, dtype=np.uint8) * 255  # Initialize mask

        for template_file in template_files:
            img1 = cv.imread(template_file)
            if img1 is None:
                logging.warning(f"Failed to load template image {template_file}")
                continue

            img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            kp1, des1 = detector.detectAndCompute(img1_gray, None)

            while True:
                kp2, des2 = detector.detectAndCompute(img2_gray, mask)
                if des1 is None or des2 is None:
                    logging.info("Descriptors not found, skipping match.")
                    break

                if use_knn:
                    raw_matches = matcher.knnMatch(des1, des2, k=2)
                    matches = [m for m, n in raw_matches if m.distance < 0.7 * n.distance] if ratio_test else [m[0] for
                                                                                                               m in
                                                                                                               raw_matches]
                else:
                    matches = matcher.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                logging.info(f"Number of matches: {len(matches)}")
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, mask_homography = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    logging.info(f"Homography matrix: {M}")
                    if M is not None and mask_homography.sum() > 10:
                        h, w = img1.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv.perspectiveTransform(pts, M)
                        img2_color = cv.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)

                        for pt in np.int32(dst):
                            cv.circle(img2_color, tuple(pt[0]), 10, (0, 0, 255), -1)  # Larger circle

                        cv.fillConvexPoly(mask, np.int32(dst), 0)
                        logging.info("Match found and mask updated.")
                    else:
                        logging.info("No valid homography found.")
                        break  # No good homography found
                else:
                    logging.info("Not enough matches.")
                    break  # Not enough matches

            if len(matches) > 10:
                # Draw thick matches
                img_matches = draw_thick_matches(img1, kp1, img2_color, kp2, matches, color=(0, 255, 0), thickness=3)
                output_file = os.path.join(method_dir,
                                           f"{os.path.basename(train_file)}_{os.path.basename(template_file)}_matches.png")
                cv.imwrite(output_file, img_matches)
                logging.info(f"Match visualization saved: {output_file}")


# Initialize detectors and matchers
sift = cv.SIFT_create()
bf_sift = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Process images with different methods
process_and_save('sift_bf', sift, bf_sift, train_directory, template_directory, output_directory, use_knn=True,
                 ratio_test=True)
process_and_save('sift_flann', sift, flann, train_directory, template_directory, output_directory, use_knn=True,
                 ratio_test=True)
