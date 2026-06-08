# import libraries
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Iterable
import cv2
import numpy as np
import math
import importlib.util
import sys
import plotly.graph_objects as go
import open3d as o3d
from scipy.optimize import least_squares


def load_week2_module(week2_dir: Path):
    """
    Load the completed Week 2 sfm_utils.py by path.
    """

    module_path = Path(week2_dir) / "sfm_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 2 sfm_utils.py: {module_path}")
    spec = importlib.util.spec_from_file_location("week2_sfm_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 2 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def load_week3_module(week3_dir: Path):
    """
    Load the completed Week 3 sfm_utils.py by path.
    """

    module_path = Path(week3_dir) / "two_view_utils.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find Week 3 sfm_utils.py: {module_path}")
    spec = importlib.util.spec_from_file_location("week3_sfm_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Week 3 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def triangulate_points_new(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
) -> np.ndarray:
    """
    This function is almost same as week3 function: triangulate_points.
    However, week3 function assumes image 1 is origin.
    This function is more generalized.
    """

    m1 = np.concatenate((R1, t1), axis = 1)
    m2 = np.concatenate((R2, t2), axis = 1)
    # calculate target projection matrix
    p1 = K @ m1
    p2 = K @ m2
    # calculate homogeneous 4D points
    # remember shape of points output by this function is (4,N)
    p4 = cv2.triangulatePoints(p1, p2, pts1.T, pts2.T)
    # p4 = p4.reshape((p4.shape[0], p4.shape[2]))
    # get 3D points
    p3 = p4[:3] / p4[3] # (3, N)
    p3 = p3.T # (N, 3)
    return p3

def build_pnp_from_match(
    image_i: int,
    image_j: int,
    matches_ij: list,
    features: list,
    points3d: np.ndarray,
    obs_to_point: dict,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    This function is used to filter matches between image i and j for PnP pose estimation.
    Feature_i mush have contributed to one 3D point.
    Each feature_j must be used only once.
    """

    # create empty list to store result
    pnp_points3d = []
    pnp_pts2d = []
    point_ids = []
    keypoint_indices_j = []
    used_keypoints_j = set()
    # iterate over all matches
    for match in matches_ij:
        kp_i = int(match.queryIdx)
        kp_j = int(match.trainIdx)
        obs_i = (image_i, kp_i)
        obs_j = (image_j, kp_j)
        # feature_i must have a corresponding 3D point
        if obs_i not in obs_to_point:
            continue
        # each feature_j should just be used once
        if obs_j in used_keypoints_j:
            continue
        # for those matches passing the tests, store them
        point_id = obs_to_point[obs_i]
        point3d = points3d[point_id]
        pt2d = features[image_j].keypoints[kp_j].pt
        pnp_points3d.append(point3d)
        pnp_pts2d.append(pt2d)
        point_ids.append(point_id)
        keypoint_indices_j.append(kp_j)
        used_keypoints_j.add(obs_j)
    # if no match passes test, return empty result
    if len(pnp_points3d) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
            [],
        )
    # return result
    return (
        np.asarray(pnp_points3d, dtype=np.float64),
        np.asarray(pnp_pts2d, dtype=np.float64),
        point_ids,
        keypoint_indices_j,
    )


def merge_pnp_chunks(
    chunks: list[tuple[np.ndarray, np.ndarray, list[int], list[int]]],
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    This function filters matches between image j and all registered images.
    Each feature j should justbbe used once.
    If feture j is macthed to more than one feature from other images.
    There are two possibilities.
    1. Features from other images represent same 3D point so that keeping one is enough.
    2. If they represent different features, they might be wrong so we just keep one. (we have further filtering later)
    """
    
    # create empty lists to store result
    merged_points3d = []
    merged_pts2d = []
    merged_point_ids = []
    merged_kp_indices_j = []
    used_kp_indices_j = set()
    # iterate overall chunks
    for points3d_chunk, pts2d_chunk, point_ids_chunk, kp_indices_j_chunk in chunks:
        for point3d, pt2d, point_id, kp_j in zip(
            points3d_chunk,
            pts2d_chunk,
            point_ids_chunk,
            kp_indices_j_chunk,
        ):
            kp_j = int(kp_j)
            # skip this chunk if it is already used
            if kp_j in used_kp_indices_j:
                continue
            # store result
            used_kp_indices_j.add(kp_j)
            merged_points3d.append(point3d)
            merged_pts2d.append(pt2d)
            merged_point_ids.append(point_id)
            merged_kp_indices_j.append(kp_j)
    # if no point is left, return empty list
    if len(merged_points3d) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
            [],
        )
    # return result
    return (
        np.asarray(merged_points3d, dtype=np.float64),
        np.asarray(merged_pts2d, dtype=np.float64),
        merged_point_ids,
        merged_kp_indices_j,
    )

def collect_new_point_matches(
    image_i: int,
    image_j: int,
    matches_ij: list,
    features: list,
    obs_to_point: dict,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    Filter matches for triangulating new 3D points.
    Both features from image i and j should have no corresponding 3D point.
    Avoid one to many matches and many to one macthes.
    """

    # create empty lists to store results
    pts_i = []
    pts_j = []
    kp_indices_i = []
    kp_indices_j = []
    used_kp_i = set()
    used_kp_j = set()
    # iterate overall matches
    for match in matches_ij:
        kp_i = int(match.queryIdx)
        kp_j = int(match.trainIdx)
        obs_i = (image_i, kp_i)
        obs_j = (image_j, kp_j)
        # only save those matches without 3D point
        if obs_i in obs_to_point or obs_j in obs_to_point:
            continue
        # avoid one to many or many to one
        if kp_i in used_kp_i or kp_j in used_kp_j:
            continue
        # save result
        pt_i = features[image_i].keypoints[kp_i].pt
        pt_j = features[image_j].keypoints[kp_j].pt
        pts_i.append(pt_i)
        pts_j.append(pt_j)
        kp_indices_i.append(kp_i)
        kp_indices_j.append(kp_j)
        used_kp_i.add(kp_i)
        used_kp_j.add(kp_j)
    # return empty list if no features left
    if len(pts_i) == 0:
        return (
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            [],
            [],
        )
    # return result
    return (
        np.asarray(pts_i, dtype=np.float64),
        np.asarray(pts_j, dtype=np.float64),
        kp_indices_i,
        kp_indices_j,
    )

def append_new_points(
    points3d: np.ndarray,
    colours: np.ndarray,
    tracks: list[dict[int, int]],
    obs_to_point: dict,
    image_i: int,
    image_j: int,
    new_points3d: np.ndarray,
    new_colours: np.ndarray,
    kp_indices_i: list[int],
    kp_indices_j: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add new points to total 3D points array.
    Check input data have consistent dimension.
    """
 
    # if there is no new 3D points, just return total 3D points
    if len(new_points3d) == 0:
        return points3d, colours
    # reshape array into (N, 3) to avoid dimension error due to one dimension input
    new_points3d = np.asarray(new_points3d, dtype=np.float64).reshape(-1, 3)
    new_colours = np.asarray(new_colours, dtype=np.uint8).reshape(-1, 3)
    # check dimension consistence
    if len(new_points3d) != len(kp_indices_i) or len(new_points3d) != len(kp_indices_j):
        raise ValueError("new_points3d and keypoint index lists must have the same length")
    start_point_id = len(tracks)
    # start adding data
    for offset, (kp_i, kp_j) in enumerate(zip(kp_indices_i, kp_indices_j)):
        # begin from end of existing array
        point_id = start_point_id + offset
        # update tracks and obs_to_point
        tracks.append({
            image_i: int(kp_i),
            image_j: int(kp_j),
        })
        obs_to_point[(image_i, int(kp_i))] = point_id
        obs_to_point[(image_j, int(kp_j))] = point_id
    if len(points3d) == 0:
        points3d = new_points3d.copy()
    else:
        # stack 3D point array: (N, 3) -> (N+k, 3)
        points3d = np.vstack([points3d, new_points3d])
    if len(colours) == 0:
        colours = new_colours.copy()
    else:
        # stack color array: (N, 3) -> (N+k, 3)
        colours = np.vstack([colours, new_colours])
    return points3d, colours

def filter_reconstructed_points_new(
    points3d: np.ndarray,
    errors1: np.ndarray,
    errors2: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    week3,
    max_reprojection_error: float = 4.0,
) -> np.ndarray:
    """
    This function is almost same as week3 function: filter_reconstructed_points.
    However, week3 function assumes image 1 is origin.
    This function is more generalized.
    """

    # get mask for finite 3D coordinate
    mask_coordinate = np.isfinite(points3d) # (N, 3)
    mask_coordinate = mask_coordinate[:, 0] & mask_coordinate[:, 1] & mask_coordinate[:, 2]
    # get mask for depth in both cameras
    _, d1 = week3.compute_depths(points3d, R1, t1)
    _, d2 = week3.compute_depths(points3d, R2, t2)
    mask_depth = (d1 > 0) & (d2 > 0)
    # get mask for reprojection error
    mask_error = (errors1 < max_reprojection_error) & (errors2 < max_reprojection_error)
    # combine three masks
    result = mask_coordinate & mask_depth & mask_error
    return result

def select_initial_pair_by_matches(
    features: list,
    week2,
    ratio: float,
    max_candidates: int | None, # only choose in first max_candidates images if this parameter is set
    max_pair_gap: int | None, # initial pair can only have a maximum gap of max_pair_gap if this parameter is set
) -> tuple[int, int, dict]:
    """
    Select the initial image pair using only the number of descriptor matches.
    Make sure the selection is under limitation: max_candidates and max_pair_gap.
    """

    # only choose in first max_candidates images if this parameter is set
    n_total = len(features)
    n_use = n_total if max_candidates is None else min(n_total, max_candidates)
    best_i = None
    best_j = None
    best_match_count = -1
    for i in range(n_use):
        for j in range(i + 1, n_use):
            # initial pair can only have a maximum gap of max_pair_gap if this parameter is set
            if max_pair_gap is not None and (j - i) > max_pair_gap:
                continue
            # get matches between image i and j
            matches = week2.match_descriptors(
                features[i].descriptors,
                features[j].descriptors,
                ratio=ratio,
            )
            # calculate number of matches
            match_count = len(matches)
            # update best pair
            if match_count > best_match_count:
                best_match_count = match_count
                best_i = i
                best_j = j
    # report error if best pair is not found
    if best_i is None or best_j is None:
        raise ValueError("Could not find a valid initial pair.")
    # return result
    return best_i, best_j, {
        "matches": best_match_count,
    }

def select_next_image_by_matches(
    registered_images: set[int],
    unregistered_images: set[int],
    features: list,
    week2,
    ratio: float,
) -> tuple[int | None, int | None]:
    """
    Select the next image using the total number of descriptor matches against all currently registered images.
    """

    best_j = None
    best_match_count = -1
    # iterate overall unregistered images
    for j in sorted(unregistered_images):
        total_matches = 0
        # iterate overall registered images
        for i in sorted(registered_images):
            # get matches
            matches_ij = week2.match_descriptors(
                features[i].descriptors,
                features[j].descriptors,
                ratio=ratio,
            )
            # calculate number of matches
            match_count = len(matches_ij)
            total_matches += match_count
        # update best 
        if total_matches > best_match_count:
            best_match_count = total_matches
            best_j = j
    return best_j, total_matches

#---------------------------------------------------------------------------------------------------------
# following functions are all used for final filtering
def filter_points_by_track_length(
    tracks: list[dict[int, int]],
    min_track_length: int = 3,
) -> np.ndarray:
    """
    Keep those points observed by at least min_track_length images.
    Return a mask.
    """

    # create zero mask
    n_points = len(tracks)
    keep_mask = np.zeros(n_points, dtype=bool)
    # filling the mask
    for point_id, track in enumerate(tracks):
        keep_mask[point_id] = len(track) >= min_track_length
    return keep_mask

def filter_points_by_reprojection(
    points3d: np.ndarray,
    tracks: list[dict[int, int]],
    camera_pose: dict[int, tuple[np.ndarray, np.ndarray]],
    features: list,
    K: np.ndarray,
    week3,
    max_mean_error: float = 5.0,
    max_max_error: float = 8.0,
) -> np.ndarray:
    """
    Keep those points under the constraint of reprojection errors.
    This filtering is different from the reprojection error filtering in triangulation.
    Previous one is just filtering between two features used to do triangulation.
    This filtering considers all features related to this 3D point.
    Return a mask.
    """

    # create zero arrays
    n_points = len(points3d)
    keep_mask = np.zeros(n_points, dtype=bool)
    mean_errors = np.full(n_points, np.inf, dtype=np.float64)
    max_errors = np.full(n_points, np.inf, dtype=np.float64)
    # iterate overall 3D points
    for point_id, track in enumerate(tracks):
        point = points3d[point_id:point_id + 1] # keep dimension (1, 3) to satisfy dimension requirment in following functions
        # create emtpy list to store reprojection errors
        errors = []
        # iterate overall features related to that 3D point
        for image_id, kp_idx in track.items():
            # get poition of feature
            observed_pt = np.asarray(
                [features[image_id].keypoints[kp_idx].pt],
                dtype=np.float64,
            )
            # get rotation and translation of that image
            R, t = camera_pose[image_id]
            # caluclate reprojection error
            err = week3.compute_reprojection_errors(
                point,
                observed_pt,
                K,
                R,
                t,
            )[0]
            # add to errors
            errors.append(float(err))
        # get mean and max error of a 3D point
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        # store mean and max error to corresponding position in mean_errors and max_errors
        mean_errors[point_id] = mean_err
        max_errors[point_id] = max_err
        # generate mask
        keep_mask[point_id] = (
            mean_err <= max_mean_error and
            max_err <= max_max_error
        )
    return keep_mask

def filter_points_by_spatial_outlier(
    points3d: np.ndarray,
    keep_percentile: float = 98.0,
) -> np.ndarray:
    """
    Remove those points far away from reconstruction center.
    """

    # if there is no 3D point, return empty result
    n_points = len(points3d)
    if n_points == 0:
        return np.empty((0,), dtype=bool), np.empty((0,), dtype=np.float64)
    # calculate reconstruction center
    center = np.median(points3d, axis=0)
    # calculate l2 distance of each 3D point to center
    distances = np.linalg.norm(points3d - center, axis=1)
    # calculate threshold calue
    threshold = np.percentile(distances, keep_percentile)
    # generate mask
    keep_mask = distances <= threshold
    return keep_mask

def apply_point_filter_mask(
    points3d: np.ndarray,
    colours: np.ndarray,
    tracks: list[dict[int, int]],
    obs_to_point: dict[tuple[int, int], int],
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[int, int]], dict[tuple[int, int], int]]:
    """
    Given a mask, generate new key data.
    """

    # get indexes of True elements
    keep_indices = np.where(keep_mask)[0]
    # get a dictionary used to update tracks and obs_to_point
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}
    # apply mask to 3D points and colors
    new_points3d = points3d[keep_indices]
    new_colours = colours[keep_indices] 
    # update tracks
    new_tracks = []
    for old_idx in keep_indices:
        new_tracks.append(dict(tracks[old_idx]))
    # update obs_to_point
    new_obs_to_point = {}
    for obs, old_point_id in obs_to_point.items():
        if old_point_id in old_to_new:
            new_obs_to_point[obs] = old_to_new[old_point_id]
    return new_points3d, new_colours, new_tracks, new_obs_to_point

def prune_bad_points(
    points3d: np.ndarray,
    colours: np.ndarray,
    tracks: list[dict[int, int]],
    obs_to_point: dict[tuple[int, int], int],
    camera_pose: dict[int, tuple[np.ndarray, np.ndarray]],
    features: list,
    K: np.ndarray,
    week3,
    min_track_length: int = 3,
    max_mean_error: float = 5.0,
    max_max_error: float = 8.0,
    spatial_keep_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[int, int]], dict[tuple[int, int], int], int]:
    """
    Apply three stage filtering.
    1. track length filtering
    2. reprojection error filtering
    3. spatial filtering
    """

    # if there is no point, return 0
    if len(points3d) == 0:
        return points3d, colours, tracks, obs_to_point, len(points3d)
    # get track length filtering mask
    mask_track = filter_points_by_track_length(
        tracks,
        min_track_length=min_track_length,
    )
    # get reprojection error filtering mask
    mask_reproj = filter_points_by_reprojection(
        points3d,
        tracks,
        camera_pose,
        features,
        K,
        week3,
        max_mean_error=max_mean_error,
        max_max_error=max_max_error,
    )
    # get spatial mask
    mask_spatial = filter_points_by_spatial_outlier(
        points3d,
        keep_percentile=spatial_keep_percentile,
    )
    # combine three masks
    final_mask = mask_track & mask_reproj & mask_spatial
    # apply final mask
    new_points3d, new_colours, new_tracks, new_obs_to_point = apply_point_filter_mask(
        points3d,
        colours,
        tracks,
        obs_to_point,
        final_mask,
    )
    return new_points3d, new_colours, new_tracks, new_obs_to_point, len(points3d) 
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
# the following functions are used for optimization
# optimization used here is Bundle Adjustment
# key idea is fine-tuning 3D coordinates and camera pose to get smaller reprojection error
# optimizer is non-liear least square method
def _pose_to_rtvec(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3), t.reshape(3)

def _rtvec_to_pose(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    t = tvec.reshape(3, 1)
    return R, t

def _project_points_cv(
    points3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    proj, _ = cv2.projectPoints(
        points3d.reshape(-1, 1, 3),
        rvec.reshape(3, 1),
        tvec.reshape(3, 1),
        K,
        None,
    )
    return proj.reshape(-1, 2)

def run_bundle_adjustment(
    camera_pose: dict[int, tuple[np.ndarray, np.ndarray]],
    points3d: np.ndarray,
    tracks: list[dict[int, int]],
    features: list,
    K: np.ndarray,
    camera_ids: list[int] | None = None,
    fixed_camera_id: int | None = None,
    min_track_observations: int = 2,
    robust_loss: str = "huber",
    f_scale: float = 2.0,
    max_nfev: int = 10,
    verbose: int = 1,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray, dict]:
    """
    Bundle adjustment for the selected cameras and the 3D points they observe.
    Args:
        camera_pose: {image_id: (R, t)}
        points3d: Nx3 global point cloud
        tracks: tracks[point_id] = {image_id: kp_idx}
        features: list of image feature objects
        K: 3x3 intrinsic matrix
        camera_ids: cameras included in BA; default = all registered cameras
        fixed_camera_id: one camera kept fixed to remove gauge freedom
        min_track_observations: only optimize points seen at least this many times
        robust_loss: "linear", "soft_l1", "huber", ...
        f_scale: robust loss scale
        max_nfev: optimizer iterations
        verbose: scipy least_squares verbosity
    Returns:
        new_camera_pose, new_points3d, stats
    """
    if camera_ids is None:
        camera_ids = sorted(camera_pose.keys())
    else:
        camera_ids = sorted([cid for cid in camera_ids if cid in camera_pose])
    if len(camera_ids) < 2 or len(points3d) == 0:
        stats = {
            "success": False,
            "reason": "not enough cameras or points",
            "num_cameras": len(camera_ids),
            "num_points": len(points3d),
            "num_observations": 0,
        }
        return camera_pose, points3d, stats
    if fixed_camera_id is None:
        fixed_camera_id = camera_ids[0]
    if fixed_camera_id not in camera_ids:
        raise ValueError("fixed_camera_id must be included in camera_ids")
    variable_camera_ids = [cid for cid in camera_ids if cid != fixed_camera_id]
    cam_id_to_var_idx = {cid: idx for idx, cid in enumerate(variable_camera_ids)}
    # Collect points and observations used in BA
    selected_point_ids = []
    point_id_to_local = {}
    observations = []  # (image_id, local_point_idx, observed_pt)
    for point_id, track in enumerate(tracks):
        obs_for_this_point = []
        for image_id, kp_idx in track.items():
            if image_id not in camera_ids:
                continue
            observed_pt = np.asarray(
                features[image_id].keypoints[kp_idx].pt,
                dtype=np.float64,
            )
            obs_for_this_point.append((image_id, observed_pt))
        if len(obs_for_this_point) < min_track_observations:
            continue
        local_point_idx = len(selected_point_ids)
        selected_point_ids.append(point_id)
        point_id_to_local[point_id] = local_point_idx
        for image_id, observed_pt in obs_for_this_point:
            observations.append((image_id, local_point_idx, observed_pt))
    if len(selected_point_ids) == 0 or len(observations) == 0:
        stats = {
            "success": False,
            "reason": "no valid BA points/observations",
            "num_cameras": len(camera_ids),
            "num_points": 0,
            "num_observations": 0,
        }
        return camera_pose, points3d, stats
    # Initial camera parameters
    cam_params0 = []
    for cid in variable_camera_ids:
        R, t = camera_pose[cid]
        rvec, tvec = _pose_to_rtvec(R, t)
        cam_params0.append(np.hstack([rvec, tvec]))
    if len(cam_params0) > 0:
        cam_params0 = np.asarray(cam_params0, dtype=np.float64).reshape(-1, 6)
        cam_params0_flat = cam_params0.ravel()
    else:
        cam_params0_flat = np.empty((0,), dtype=np.float64)
    # Fixed camera params
    fixed_R, fixed_t = camera_pose[fixed_camera_id]
    fixed_rvec, fixed_tvec = _pose_to_rtvec(fixed_R, fixed_t)
    # Initial point parameters
    pts0 = points3d[selected_point_ids].astype(np.float64).reshape(-1, 3)
    pts0_flat = pts0.ravel()
    x0 = np.hstack([cam_params0_flat, pts0_flat])
    n_var_cams = len(variable_camera_ids)
    n_sel_points = len(selected_point_ids)
    cam_block_size = 6 * n_var_cams
    def residual_fn(x: np.ndarray) -> np.ndarray:
        if n_var_cams > 0:
            cam_params = x[:cam_block_size].reshape(n_var_cams, 6)
        else:
            cam_params = np.empty((0, 6), dtype=np.float64)
        pts = x[cam_block_size:].reshape(n_sel_points, 3)
        residuals = np.zeros((2 * len(observations),), dtype=np.float64)
        for obs_idx, (image_id, local_point_idx, observed_pt) in enumerate(observations):
            if image_id == fixed_camera_id:
                rvec = fixed_rvec
                tvec = fixed_tvec
            else:
                var_idx = cam_id_to_var_idx[image_id]
                rvec = cam_params[var_idx, :3]
                tvec = cam_params[var_idx, 3:]
            proj = _project_points_cv(
                pts[local_point_idx:local_point_idx + 1],
                rvec,
                tvec,
                K,
            )[0]
            residuals[2 * obs_idx: 2 * obs_idx + 2] = proj - observed_pt
        return residuals
    result = least_squares(
        residual_fn,
        x0,
        loss=robust_loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        verbose=verbose,
    )
    x_opt = result.x
    if n_var_cams > 0:
        cam_params_opt = x_opt[:cam_block_size].reshape(n_var_cams, 6)
    else:
        cam_params_opt = np.empty((0, 6), dtype=np.float64)
    pts_opt = x_opt[cam_block_size:].reshape(n_sel_points, 3)
    new_camera_pose = dict(camera_pose)
    for cid in variable_camera_ids:
        var_idx = cam_id_to_var_idx[cid]
        rvec = cam_params_opt[var_idx, :3]
        tvec = cam_params_opt[var_idx, 3:]
        new_camera_pose[cid] = _rtvec_to_pose(rvec, tvec)
    new_points3d = points3d.copy()
    for old_point_id, local_point_idx in point_id_to_local.items():
        new_points3d[old_point_id] = pts_opt[local_point_idx]
    initial_rmse = np.sqrt(np.mean(residual_fn(x0) ** 2))
    final_rmse = np.sqrt(np.mean(residual_fn(x_opt) ** 2))
    stats = {
        "success": bool(result.success),
        "message": result.message,
        "num_cameras": len(camera_ids),
        "num_variable_cameras": len(variable_camera_ids),
        "num_points": len(selected_point_ids),
        "num_observations": len(observations),
        "initial_rmse": float(initial_rmse),
        "final_rmse": float(final_rmse),
        "cost": float(result.cost),
        "nfev": int(result.nfev),
    }
    return new_camera_pose, new_points3d, stats
#---------------------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------------------
# following functions are all used for visualization only
def camera_center(R, t):
    return (-R.T @ t.reshape(3, 1)).ravel()

def camera_frustum(R, t, scale=0.5):
    center = camera_center(R, t)
    corners_cam = np.array([
        [-0.5, -0.35, 1.0],
        [ 0.5, -0.35, 1.0],
        [ 0.5,  0.35, 1.0],
        [-0.5,  0.35, 1.0],
    ], dtype=np.float64)
    # camera-to-world rotation
    R_wc = R.T
    corners_world = center[None, :] + scale * (R_wc @ corners_cam.T).T
    return center, corners_world

def plotly_multi_view_scene(points3d, colours, camera_poses, output_html):
    fig = go.Figure()
    # point cloud
    if len(points3d) > 0:
        if len(colours) == len(points3d):
            rgb_strings = [
                f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
                for c in colours
            ]
        else:
            rgb_strings = "royalblue"
        fig.add_trace(go.Scatter3d(
            x=points3d[:, 0],
            y=points3d[:, 1],
            z=points3d[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=rgb_strings,
                opacity=0.9,
            ),
            name="Points",
        ))
    centers = []
    camera_colors = ["#2458a6", "#a33b3b", "#2f7d32", "#7a4ea3", "#b46b00"]
    for idx, (label, R, t) in enumerate(camera_poses):
        color = camera_colors[idx % len(camera_colors)]
        center, frustum = camera_frustum(R, t, scale=0.6)
        centers.append(center)
        # camera center
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode="markers+text",
            marker=dict(size=6, color=color),
            text=[label],
            textposition="top center",
            name=label,
        ))
        # frustum edges: center -> each corner
        for corner in frustum:
            fig.add_trace(go.Scatter3d(
                x=[center[0], corner[0]],
                y=[center[1], corner[1]],
                z=[center[2], corner[2]],
                mode="lines",
                line=dict(color=color, width=4),
                showlegend=False,
            ))
        # front rectangle
        order = [0, 1, 2, 3, 0]
        fig.add_trace(go.Scatter3d(
            x=frustum[order, 0],
            y=frustum[order, 1],
            z=frustum[order, 2],
            mode="lines",
            line=dict(color=color, width=4),
            showlegend=False,
        ))
    # camera trajectory
    if len(centers) >= 2:
        centers = np.asarray(centers)
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode="lines",
            line=dict(color="gray", width=5),
            name="Camera path",
        ))
    fig.update_layout(
        title="Multi-view Sparse Reconstruction",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="white",
        ),
        template="plotly_white",
        width=1100,
        height=800,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.write_html(str(output_html))

def make_camera_lineset(R, t, scale=0.5, color=(1.0, 0.2, 0.2)):
    center = camera_center(R, t)
    corners_cam = np.array([
        [-0.5, -0.35, 1.0],
        [ 0.5, -0.35, 1.0],
        [ 0.5,  0.35, 1.0],
        [-0.5,  0.35, 1.0],
    ], dtype=np.float64)
    R_wc = R.T
    corners_world = center[None, :] + scale * (R_wc @ corners_cam.T).T
    points = np.vstack([center[None, :], corners_world])
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],   # center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],   # front rectangle
    ]
    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def visualize_open3d_scene(points3d, colours, camera_poses):
    geometries = []
    # point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)
    if len(colours) == len(points3d):
        pcd.colors = o3d.utility.Vector3dVector(colours.astype(np.float64) / 255.0)
    geometries.append(pcd)
    # camera frustums
    camera_colors = [
        (0.14, 0.35, 0.65),
        (0.64, 0.23, 0.23),
        (0.18, 0.49, 0.20),
        (0.48, 0.31, 0.64),
        (0.70, 0.42, 0.00),
    ]
    centers = []
    for idx, (label, R, t) in enumerate(camera_poses):
        color = camera_colors[idx % len(camera_colors)]
        cam = make_camera_lineset(R, t, scale=0.6, color=color)
        geometries.append(cam)
        centers.append(camera_center(R, t))
    # camera trajectory
    if len(centers) >= 2:
        traj = o3d.geometry.LineSet()
        traj.points = o3d.utility.Vector3dVector(np.asarray(centers))
        traj.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(len(centers) - 1)]
        )
        traj.colors = o3d.utility.Vector3dVector(
            [[0.5, 0.5, 0.5] for _ in range(len(centers) - 1)]
        )
        geometries.append(traj)
    o3d.visualization.draw_geometries(
        geometries,
        window_name="GF4 Multi-view Reconstruction",
        width=1280,
        height=900,
    )
#---------------------------------------------------------------------------------------------------------