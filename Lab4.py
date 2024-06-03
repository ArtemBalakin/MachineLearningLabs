import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_acceptable_transform(transformed_corners, max_rotation_deg=20):
    transformed_corners = transformed_corners.squeeze()
    sides = np.diff(np.vstack([transformed_corners, transformed_corners[0]]), axis=0)
    lengths = np.linalg.norm(sides, axis=1)
    cos_angles = [(sides[i] @ sides[(i + 1) % 4]) / (lengths[i] * lengths[(i + 1) % 4]) for i in range(4)]
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi
    acceptable = np.all((angles > 90 - max_rotation_deg) & (angles < 90 + max_rotation_deg))
    logging.debug(f"Transformed corners acceptable: {acceptable}")
    return acceptable

def find_matching_boxes(image, template, params):
    MAX_MATCHING_OBJECTS = params.get('max_matching_objects', 10)
    SIFT_DISTANCE_THRESHOLD = params.get('SIFT_distance_threshold', 0.5)

    detector = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    keypoints2, descriptors2 = detector.detectAndCompute(template, None)
    matched_boxes = []
    matching_img = image.copy()

    logging.info(f"Starting to find matching boxes for template.")
    for i in range(MAX_MATCHING_OBJECTS):
        keypoints1, descriptors1 = detector.detectAndCompute(matching_img, None)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = [m for m, n in matches if m.distance < SIFT_DISTANCE_THRESHOLD * n.distance]

        logging.info(f"Found {len(good_matches)} good matches.")
        if not good_matches:
            logging.info("No more matching boxes found.")
            break

        distances = [m.distance for m in good_matches]
        if distances:
            threshold_distance = np.median(distances) * 0.75
            good_matches = [m for m in good_matches if m.distance <= threshold_distance]

        if len(good_matches) < 4:
            logging.warning("Not enough matches to compute homography.")
            continue

        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

        try:
            H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 2)
            if H is None:
                logging.error("Homography matrix was not found.")
                continue

            h, w = template.shape[:2]
            corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)

            if not is_acceptable_transform(transformed_corners):
                continue

            matched_boxes.append(transformed_corners)
            matching_img2 = cv2.cvtColor(matching_img, cv2.COLOR_BGR2GRAY)
            mask = np.ones_like(matching_img2) * 255
            cv2.fillPoly(mask, [np.int32(transformed_corners)], 0)
            mask = cv2.bitwise_not(mask)
            matching_img = cv2.inpaint(matching_img, mask, 3, cv2.INPAINT_TELEA)

        except cv2.error as e:
            logging.error(f"Error during homography computation or transformation: {str(e)}")

    return matched_boxes

train_image_dir = 'input/lab3/train'
template_dir = 'input/lab3'
params = {
    'max_matching_objects': 20,
    'SIFT_distance_threshold': 0.9
}

for image_filename in os.listdir(train_image_dir):
    if image_filename.endswith('.jpg'):
        img_path = os.path.join(train_image_dir, image_filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logging.info(f"Processing image {img_path}")

        for template_filename in os.listdir(template_dir):
            if template_filename.endswith('.jpg'):
                template_path = os.path.join(template_dir, template_filename)
                template = cv2.imread(template_path)
                template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
                logging.info(f"Processing template {template_path}")

                matched_boxes = find_matching_boxes(img, template, params)

                for box in matched_boxes:
                    cv2.polylines(img, [np.int32(box)], True, (0, 255, 0), 3, cv2.LINE_AA)

        plt.imshow(img)
        plt.show()
