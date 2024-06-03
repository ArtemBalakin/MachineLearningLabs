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

# Clear the output directory
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.makedirs(output_directory)


def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))


def pad_images(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 < h2:
        img1 = cv.copyMakeBorder(img1, 0, h2 - h1, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img2 = cv.copyMakeBorder(img2, 0, h1 - h2, 0, 0, cv.BORDER_CONSTANT, value=[0, 0, 0])
    return img1, img2


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

        for template_file in template_files:
            img1 = cv.imread(template_file)
            if img1 is None:
                logging.warning(f"Failed to load template image {template_file}")
                continue

            img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            img2 = original_img.copy()
            img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

            kp1, des1 = detector.detectAndCompute(img1_gray, None)
            kp2, des2 = detector.detectAndCompute(img2_gray, None)

            if des1 is None or des2 is None:
                logging.info("Descriptors not found, skipping match.")
                continue

            if use_knn:
                raw_matches = matcher.knnMatch(des1, des2, k=2)
                matches = [m for m, n in raw_matches if m.distance < 0.75 * n.distance] if ratio_test else [m[0] for m
                                                                                                            in
                                                                                                            raw_matches]
            else:
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

            for i in range(5):
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                    if M is not None:
                        h, w = img1.shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv.perspectiveTransform(pts, M)
                        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
                        logging.info(f"Match {i + 1}: Homography found, updating image.")

                        # Create mask for the area where homography was found
                        mask = np.zeros(img2_gray.shape, dtype=np.uint8)
                        cv.fillConvexPoly(mask, np.int32(dst), 255)
                        img2_gray = cv.bitwise_and(img2_gray, img2_gray, mask=cv.bitwise_not(mask))

                        # Re-compute keypoints and descriptors for the next iteration
                        kp2, des2 = detector.detectAndCompute(img2_gray, None)
                        if des2 is None:
                            break
                        matches = matcher.match(des1, des2)
                        matches = sorted(matches, key=lambda x: x.distance)
                    else:
                        break
                else:
                    break

            img1, img2 = pad_images(img1, img2)
            if img1 is not None and img2 is not None:
                combined_img = np.concatenate((img1, img2), axis=1)
                output_file = os.path.join(method_dir,
                                           f"{os.path.basename(train_file)}_{os.path.basename(template_file)}.png")
                cv.imwrite(output_file, combined_img)
                logging.info(f"Result saved: {output_file}")


# Initialize detectors and matchers
sift = cv.SIFT_create()
bf_sift = cv.BFMatcher()
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

# Process images
process_and_save('sift_bf', sift, bf_sift, train_directory, template_directory, output_directory, use_knn=True,
                 ratio_test=True)
process_and_save('sift_flann', sift, flann, train_directory, template_directory, output_directory, use_knn=True,
                 ratio_test=True)


