import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_and_split_image_with_overlap(image_path, N, overlap=10):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    print(height, width)
    part_width = width // N
    return [image[:, max(0, i * part_width - overlap):min(width, (i + 1) * part_width + overlap)] for i in range(N)]


def detect_features(image):
    sift = cv2.SIFT_create(nfeatures=99999999)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.003 * n.distance]  # Adjusted threshold for practicality
    return good_matches


def stitch_images(img1, img2, kp1, kp2, matches):
    if len(matches) > 10:
        # Extract location of good matches
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Find homography
        H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
        if H is not None:
            # Obtain size of both images
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            # Calculate corners of img1
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            # Calculate expected corners of img2 in the stitched image using homography
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            corners2_transformed = cv2.perspectiveTransform(corners2, H)

            # Find the combined bounding box of both images
            all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
            xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
            xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

            # Adjust the translation matrix to avoid going out of bounds
            trans_dist = [-xmin, -ymin]
            trans_mat = np.array([[1, 0, trans_dist[0]], [0, 1, trans_dist[1]], [0, 0, 1]])

            # Warp img2 to img1's plane
            width = int(xmax - xmin)
            height = int(ymax - ymin)
            warp_img = cv2.warpPerspective(img2, trans_mat.dot(H), (width, height))

            # Place img1 in the translated image
            x_start = max(-xmin, 0)
            y_start = max(-ymin, 0)
            x_end = x_start + w1
            y_end = y_start + h1

            # Check if the coordinates are within the bounds of the new image
            if y_end > y_start and x_end > x_start and y_end <= height and x_end <= width:
                warp_img[y_start:y_end, x_start:x_end] = img1

            return warp_img

    return None


image_parts = load_and_split_image_with_overlap('imagesForLab3/tsah7c9evnal289z5fig.jpg', 250, 30)
stitched_image = image_parts[0]
kp_stitched, desc_stitched = detect_features(stitched_image)

for i in range(1, len(image_parts)):
    kp_next, desc_next = detect_features(image_parts[i])
    matches = match_features(desc_stitched, desc_next)
    result = stitch_images(stitched_image, image_parts[i], kp_stitched, kp_next, matches)
    if result is not None:
        stitched_image = result
        kp_stitched, desc_stitched = detect_features(stitched_image)
        print(f"Successfully stitched part with part {i + 1}")
    else:
        print(f"Failed to stitch part with part {i + 1}")

if stitched_image is not None:
    plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
    plt.title('Final Stitched Image')
    plt.show()
else:
    print("No successful stitchings were achieved.")
