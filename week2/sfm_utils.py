"""Utility functions for GF4 Week 2 pairwise SfM front-end.

This file is intentionally a starter scaffold. Basic file handling and a few
plotting helpers are provided. The core SfM-front-end steps are marked with
TODO and should be completed by students.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageFeatures:
    """Image, keypoints, and descriptors for one input image."""

    path: Path
    image: np.ndarray
    keypoints: list[cv2.KeyPoint]
    descriptors: np.ndarray


@dataclass
class PairAnalysis:
    """Container for pairwise matching and epipolar-geometry results."""

    image_i: str
    image_j: str
    keypoints_i: int
    keypoints_j: int
    raw_matches: int
    filtered_matches: int
    ransac_inliers: int
    inlier_ratio: float
    mean_epipolar_error_all: float | None
    median_epipolar_error_all: float | None
    mean_epipolar_error_inliers: float | None
    median_epipolar_error_inliers: float | None
    max_epipolar_error_inliers: float | None
    fundamental_matrix: list[list[float]] | None

    def as_dict(self) -> dict:
        return {
            "image_i": self.image_i,
            "image_j": self.image_j,
            "keypoints_i": self.keypoints_i,
            "keypoints_j": self.keypoints_j,
            "raw_matches": self.raw_matches,
            "filtered_matches": self.filtered_matches,
            "ransac_inliers": self.ransac_inliers,
            "inlier_ratio": self.inlier_ratio,
            "mean_epipolar_error_all": self.mean_epipolar_error_all,
            "median_epipolar_error_all": self.median_epipolar_error_all,
            "mean_epipolar_error_inliers": self.mean_epipolar_error_inliers,
            "median_epipolar_error_inliers": self.median_epipolar_error_inliers,
            "max_epipolar_error_inliers": self.max_epipolar_error_inliers,
            "fundamental_matrix": self.fundamental_matrix,
        }

    def csv_dict(self) -> dict:
        """Return scalar fields suitable for CSV output."""
        data = self.as_dict()
        data["fundamental_matrix"] = (
            "" if self.fundamental_matrix is None else str(self.fundamental_matrix)
        )
        return data


def ensure_dir(path: Path) -> Path:
    """Create an output directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_image_paths(image_dir: Path, max_images: int | None = None) -> list[Path]:
    """Return sorted image paths from a directory."""
    image_dir = Path(image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = sorted(
        p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


def load_image(path: Path, max_size: int | None = None) -> np.ndarray:
    """Load an image with OpenCV in BGR order, optionally resizing the long edge."""
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if max_size is not None:
        height, width = image.shape[:2]
        scale = max_size / max(height, width)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
    return image


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    """Save a list of dictionaries as CSV."""
    rows = list(rows)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def detect_sift_features(
    image: np.ndarray,
    max_features: int = 4000,
) -> tuple[list[cv2.KeyPoint], np.ndarray]:
    """Detect SIFT keypoints and descriptors.

    TODO: Complete this function.

    Hints:
    * - Convert the image to grayscale. 
    * - Create a SIFT detector with cv2.SIFT_create(nfeatures=max_features).
    * - Return keypoints and descriptors from detector.detectAndCompute(...).
    * - If no descriptors are found, return an empty array with shape (0, 128).
    * - If OpenCV returns slightly more than max_features, keep only the first
      max_features keypoints and matching descriptor rows.
    """

    # convert image to gray
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # create sift detector
    sift = cv2.SIFT_create()
    # get keypoints and descriptors
    kp, dp = sift.detectAndCompute(gray_image, None)
    # return kp and dp after validation
    if dp is None:
        return ([],np.empty((0, 128)))
    elif len(kp) > max_features:
        return (kp[:max_features], dp[:max_features])
    else:
        return (kp, dp)

    # raise NotImplementedError("TODO: implement SIFT feature detection")


def precompute_image_features(
    image_paths: list[Path],
    max_features: int = 4000,
    max_image_size: int | None = 1600,
) -> list[ImageFeatures]:
    """Load each image and compute SIFT features once.

    TODO: Complete this function after implementing detect_sift_features.

    Dataset mode should use this function so SIFT is not recomputed for the
    same image in every pair.
    """

    # create empty list to store result
    result = []
    # iterate all images and store in result
    for i in image_paths:
        image_array = load_image(i, max_image_size)
        kp, dp = detect_sift_features(image_array, max_features)
        feature = ImageFeatures(
            path = i,
            image = image_array,
            keypoints = kp,
            descriptors = dp
        )
        result.append(feature)
    # return result
    return result

    #raise NotImplementedError("TODO: implement feature precomputation")


def raw_descriptor_matches(desc1: np.ndarray, desc2: np.ndarray) -> list[cv2.DMatch]:
    """Return one nearest-neighbour match per descriptor before Lowe filtering.

    TODO: Complete this function.

    Hints:
    * - Handle empty descriptor arrays by returning an empty list.
    * - For SIFT descriptors, use cv2.BFMatcher(cv2.NORM_L2).
    * - Use matcher.match(desc1, desc2) to get the best match in image 2 for
      each descriptor in image 1.
    * - Return the matches sorted by descriptor distance.
    """

    # match descriptors and sort them from best to least after validation
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        return sorted(bf.match(desc1,desc2), key = lambda x:x.distance) #  Ascending order from best match to least match

    # raise NotImplementedError("TODO: compute raw nearest-neighbour matches")


def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.75,
) -> list[cv2.DMatch]:
    """Match SIFT descriptors using Lowe's ratio test.

    TODO: Complete this function.

    Hints:
    * - Handle empty descriptor arrays by returning an empty list.
    * - For SIFT descriptors, use cv2.BFMatcher(cv2.NORM_L2).
    * - Use knnMatch(desc1, desc2, k=2) for the ratio test.
    - Keep a match when best_distance < ratio * second_best_distance.
    """

    # pick two best matches and do ratio test
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_match = bf.knnMatch(desc1, desc2, k=2)
        result = []
        for i, k in raw_match:
            # ratio test
            if i.distance < ratio * k.distance:
                result.append(i)
        return sorted(result, key = lambda x:x.distance) # from best match to least match

    # raise NotImplementedError("TODO: implement descriptor matching")


def count_raw_matches(desc1: np.ndarray, desc2: np.ndarray) -> int:
    """Return the number of descriptors that can be matched before filtering.

    TODO: Complete this function.

    A simple definition is len(raw_descriptor_matches(desc1, desc2)). This
    gives a useful denominator for comparing raw and filtered matching.
    """

    # count raw matches
    return len(raw_descriptor_matches(desc1, desc2))

    # raise NotImplementedError("TODO: count raw descriptor matches")


def matched_keypoint_coords(
    keypoints1: list[cv2.KeyPoint],
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert OpenCV matches into aligned Nx2 coordinate arrays.

    TODO: Complete this function.

    Remember: cv2.KeyPoint.pt is (x, y), not (row, column).
    """

    # x is column number and y is row number
    # create zero arrays
    c1 = np.zeros((len(matches),2))
    c2 = np.zeros((len(matches),2))
    # get corresponding keypoints and extract coordinates
    for i in range(len(matches)):
        c1[i,0] = keypoints1[matches[i].queryIdx].pt[0]
        c1[i,1] = keypoints1[matches[i].queryIdx].pt[1]
        c2[i,0] = keypoints2[matches[i].trainIdx].pt[0]
        c2[i,1] = keypoints2[matches[i].trainIdx].pt[1]
    return (c1, c2)

    # raise NotImplementedError("TODO: convert matches to coordinate arrays")


def estimate_fundamental_ransac(
    pts1: np.ndarray,
    pts2: np.ndarray,
    threshold: float = 1.0,
    confidence: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the fundamental matrix with OpenCV RANSAC.

    TODO: Complete this function.

    Return:
    - F: 3x3 fundamental matrix
    - inlier_mask: boolean array of shape (N,)
    """

    # get fundamental matrix and inlier matches
    F, inlier_mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold, confidence)
    # return result
    return (F, inlier_mask)

    # raise NotImplementedError("TODO: estimate fundamental matrix with RANSAC")


def compute_epipolar_errors(
    F: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """Compute point-to-epipolar-line distances in image 2.

    TODO: Complete this function.

    For each point x1 in image 1, compute the epipolar line l2 = F x1.
    Then compute the distance from the corresponding x2 to l2.
    """

    # create zero array
    result = np.zeros(pts1.shape[0])
    # calculate error for all keypoints
    for i in range(pts1.shape[0]):
        # get homogenious coordinate of x1 with last element to be 1
        x1 = np.array([pts1[i,0],pts1[i,1],1])
        # get epipolar line
        l2 = F @ x1
        # get homogenious coordinate of x2 with last element to be 1
        x2 = np.array([pts2[i,0],pts2[i,1],1])
        # calculate epipolar error
        result[i] = abs(l2[0]*x2[0] + l2[1]*x2[1] +l2[2]) / math.sqrt(l2[0]**2 + l2[1]**2)
    # return result
    return result

    # raise NotImplementedError("TODO: implement epipolar error calculation")


def draw_keypoints(
    image: np.ndarray,
    keypoints: list[cv2.KeyPoint],
    output_path: Path,
) -> None:
    """Save a keypoint visualisation."""
    ensure_dir(output_path.parent)
    vis = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(output_path), vis)


def draw_matches(
    image1: np.ndarray,
    keypoints1: list[cv2.KeyPoint],
    image2: np.ndarray,
    keypoints2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    output_path: Path,
    max_draw: int = 80,
) -> None:
    """Save a feature-match visualisation."""
    ensure_dir(output_path.parent)
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_draw]
    vis = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(str(output_path), vis)


def draw_epipolar_lines(
    image1: np.ndarray,
    image2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    F: np.ndarray,
    output_path: Path,
    max_lines: int = 20,
) -> None:
    """Save an epipolar-line visualisation.

    TODO: Complete this function.

    Hints:
    * - Sample up to max_lines corresponding points.
    * - For each x1, draw l2 = F x1 in image 2.
    * - Draw the corresponding x2 point on image 2.
    * - A simple Matplotlib figure with image1 and image2 side by side is enough.
    """

    # check validation of outout directory
    ensure_dir(output_path.parent)
    # pick up to max_lines
    if pts1.shape[0] > max_lines:
        pts1 = pts1[:max_lines]
        pts2 = pts2[:max_lines]
    # get number of column
    c = image1.shape[1]
    print(image1.shape[0])
    print(image2.shape[0])
    # create empty list
    lines = []
    # iterate all keypoints
    for i in range(len(pts1)):
        # randomly pick a color
        color = tuple(np.random.randint(0,255,3).tolist())
        # draw matches keypoints in image1 and image2
        image1 = cv2.circle(image1,tuple(map(int, pts1[i])),10,color,-1)
        image2 = cv2.circle(image2,tuple(map(int, pts2[i])),10,color,-1)
        # get homogenious coordinate of x1 with last element to be 1
        x1 = np.array([pts1[i,0],pts1[i,1],1])
        # get epipolar lines in second image
        lines = F @ x1
        # get two points in epipolar line
        x0,y0 = map(int, [0, -lines[2]/lines[1] ])
        x1,y1 = map(int, [c, -(lines[2]+lines[0]*c)/lines[1] ])
        # draw epipolar line in second image
        image2 = cv2.line(image2, (x0,y0), (x1,y1), color,1)
    # create a white gap between image1 and image2
    h = image1.shape[0]
    gap_width = 20
    gap = np.ones((h, gap_width, 3), dtype=np.uint8) * 255
    # concatenate image1, white gap and image2 inte a single image
    combined_img = np.hstack((image1, gap, image2))
    # save image to output path
    cv2.imwrite(str(output_path), combined_img)

    # raise NotImplementedError("TODO: implement epipolar-line visualisation")


def analyse_image_pair(
    image1_path: Path,
    image2_path: Path,
    output_dir: Path,
    max_features: int = 4000,
    ratio: float = 0.75,
    max_image_size: int | None = 1600,
    save_figures: bool = True,
) -> PairAnalysis:
    """Run the full Week 2 analysis for one image pair.

    TODO: Complete this function by wiring together the utilities above.

    Expected steps:
    * 1. Load both images.
    * 2. Detect SIFT features.
    * 3. Match descriptors with Lowe's ratio test.
    * 4. Convert matches to point arrays.
    * 5. Estimate F with RANSAC.
    * 6. Compute epipolar errors for all filtered matches and for RANSAC inliers.
    * 7. Save keypoint, raw-match, filtered-match, inlier, and epipolar-line figures.
    * 8. Return a PairAnalysis object.
    """

    # check validation of output directory
    ensure_dir(output_dir)
    # load image1 and image2
    image1 = load_image(image1_path, max_image_size)
    image2 = load_image(image2_path, max_image_size)
    # get features in image1 and image2
    kp1, dp1 = detect_sift_features(image1, max_features)
    kp2, dp2 = detect_sift_features(image2, max_features)
    # create ImageFeature object for image1
    f1 = ImageFeatures(
        path = image1_path,
        image = image1,
        keypoints = kp1,
        descriptors = dp1
    )
    # create ImageFeature object for image2
    f2 = ImageFeatures(
        path = image2_path,
        image = image2,
        keypoints = kp2,
        descriptors = dp2
    )
    # call analyse_feature_pair() with features of image1 and image2
    return analyse_feature_pair(f1, f2, output_dir, ratio, save_figures)

    # raise NotImplementedError("TODO: implement pair analysis pipeline")


def analyse_feature_pair(
    features1: ImageFeatures,
    features2: ImageFeatures,
    output_dir: Path,
    ratio: float = 0.75,
    save_figures: bool = True,
) -> PairAnalysis:
    """Run pair analysis using precomputed image features.

    TODO: Complete this function and call it from analyse_image_pair.

    This avoids recomputing SIFT features during all-pairs dataset analysis.
    In dataset mode, save_figures is normally False, so this function should
    return metrics without creating an output folder for every image pair.
    """

    # check validation of output directory
    ensure_dir(output_dir)
    # get image array from features
    image1 = features1.image
    image2 = features2.image
    # get keypoints and descriptors from features
    kp1 = features1.keypoints
    dp1 = features1.descriptors
    kp2 = features2.keypoints
    dp2 = features2.descriptors
    # set raw matches and filtered matches
    raw_matches = raw_descriptor_matches(dp1, dp2)
    matches = match_descriptors(dp1, dp2, ratio)
    # get coordinate of matched keypoints from both methods
    pts1_raw, pts2_raw = matched_keypoint_coords(kp1, kp2, raw_matches)
    pts1, pts2 = matched_keypoint_coords(kp1, kp2, matches)
    # get fundamental matrix and inlier mask of filtered matching
    F, inlier_mask = estimate_fundamental_ransac(pts1, pts2)
    # remove last dimension of inlier_mask and convert 0/1 to true/false
    inlier_mask = inlier_mask.squeeze() == 1
    # calculate epipolar error of all filtered matches
    ep_error = compute_epipolar_errors(F, pts1, pts2)
    # calculate epipolar error of inlier filtered matches
    ep_error_inlier = compute_epipolar_errors(F, pts1[inlier_mask], pts2[inlier_mask])
    # draw required graphs
    if save_figures == True:
        # keypoints
        draw_keypoints(image1, kp1, output_dir / "keypoints_image1.png")
        draw_keypoints(image2, kp2, output_dir / "keypoints_image2.png")

        # raw-match
        draw_matches(image1, kp1, image2, kp2, raw_matches, output_dir / "raw_match.png")

        # filtered match
        draw_matches(image1, kp1, image2, kp2, matches, output_dir / "filtered_match.png")

        # inlier filtered match
        draw_matches(image1, kp1, image2, kp2, np.array(matches)[inlier_mask].tolist(), output_dir / "inlier_match.png")

        # epipolar-line
        draw_epipolar_lines(image1, image2, pts1[inlier_mask], pts2[inlier_mask], F, output_dir / "epipolar_lines.png")
    # create result PairAnalysis result
    result = PairAnalysis(
        image_i = features1.path,
        image_j = features2.path,
        keypoints_i = len(kp1),
        keypoints_j = len(kp2),
        raw_matches = len(raw_matches),
        filtered_matches = len(matches),
        ransac_inliers = int(np.sum(inlier_mask)),
        inlier_ratio = int(np.sum(inlier_mask)) / len(matches),
        mean_epipolar_error_all = float(np.mean(ep_error)),
        median_epipolar_error_all = float(np.median(ep_error)),
        mean_epipolar_error_inliers = float(np.mean(ep_error_inlier)),
        median_epipolar_error_inliers = float(np.median(ep_error_inlier)),
        max_epipolar_error_inliers = float(np.max(ep_error_inlier)),
        fundamental_matrix = F.tolist()
    )
    return result

    # raise NotImplementedError("TODO: implement pair analysis from precomputed features")


def draw_match_graph(
    rows: list[dict],
    output_path: Path,
    min_inliers: int = 30,
) -> None:
    """Draw a match graph from pairwise metric rows.

    Edges with fewer than min_inliers are omitted to keep the graph readable.
    """
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)

    nodes = sorted({row["image_i"] for row in rows} | {row["image_j"] for row in rows})
    edges = []
    for row in rows:
        inliers = int(row["ransac_inliers"])
        if inliers >= min_inliers:
            edges.append((row["image_i"], row["image_j"], inliers))

    plt.figure(figsize=(10, 8))
    if not nodes or not edges:
        plt.text(0.5, 0.5, "No edges above threshold", ha="center", va="center")
        plt.axis("off")
    else:
        radius = 1.0
        positions = {}
        for idx, node in enumerate(nodes):
            angle = 2 * math.pi * idx / len(nodes)
            positions[node] = (radius * math.cos(angle), radius * math.sin(angle))

        max_weight = max(weight for _, _, weight in edges)
        for image_i, image_j, weight in edges:
            x1, y1 = positions[image_i]
            x2, y2 = positions[image_j]
            width = 1.0 + 4.0 * (weight / max_weight)
            plt.plot([x1, x2], [y1, y2], color="#456990", linewidth=width, alpha=0.7)
            plt.text((x1 + x2) / 2, (y1 + y2) / 2, str(weight), fontsize=7)

        for node, (x, y) in positions.items():
            plt.scatter([x], [y], s=550, color="#d8e8ff", edgecolor="#456990", zorder=3)
            label = Path(node).stem
            plt.text(x, y, label, ha="center", va="center", fontsize=8, zorder=4)

        plt.axis("off")
        plt.axis("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def select_top_initial_pairs(rows: list[dict], top_k: int = 3) -> list[dict]:
    """Select candidate Week 3 initial pairs from pairwise metrics.

    This starter version ranks by RANSAC inlier count first, then inlier ratio.
    Students should inspect the images too: the best numerical pair may have too
    little baseline for triangulation.
    """
    return sorted(
        rows,
        key=lambda row: (
            int(row["ransac_inliers"]),
            float(row["inlier_ratio"]),
            -float(row["median_epipolar_error_inliers"] or 1e9),
        ),
        reverse=True,
    )[:top_k]
