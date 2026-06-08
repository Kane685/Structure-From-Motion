# import necessary libraries
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import sys
import numpy as np

# import functions from sfm_utils
from sfm_utils import (
    load_week2_module,
    load_week3_module,
    build_pnp_from_match,   
    triangulate_points_new,
    merge_pnp_chunks,
    append_new_points,
    collect_new_point_matches,
    filter_reconstructed_points_new,
    plotly_multi_view_scene,
    visualize_open3d_scene,
    select_initial_pair_by_matches,
    select_next_image_by_matches,
    prune_bad_points,
    run_bundle_adjustment
)

# import modules from week2 and week3
DEFAULT_WEEK2_DIR = Path(__file__).resolve().parents[1] / "week2"
DEFAULT_WEEK3_DIR = Path(__file__).resolve().parents[1] / "week3"

# set input parameters
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GF4 Week 4 sparse reconstruction pipeline."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where metrics and figures will be written.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images for dataset analysis.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to load .",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe ratio-test threshold passed to the Week 2 matcher.",
    )
    parser.add_argument(
        "--focal-length-px",
        type=float,
        default=2912,
        help="Optional focal length in pixels. Default is 2912.",
    )
    parser.add_argument(
        "--principal-point",
        nargs=2,
        type=float,
        metavar=("CX", "CY"),
        default=None,
        help="Optional principal point in pixels.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.0,
        help="RANSAC threshold in pixels for essential matrix estimation.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.999,
        help="RANSAC confidence for essential matrix estimation.",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=4.0,
        help="Maximum reprojection error in pixels for keeping triangulated points.",
    )
    parser.add_argument(
        "--week2-dir",
        type=Path,
        default=DEFAULT_WEEK2_DIR,
        help="Directory containing the completed Week 2 sfm_utils.py.",
    )
    parser.add_argument(
        "--week3-dir",
        type=Path,
        default=DEFAULT_WEEK3_DIR,
        help="Directory containing the completed Week 2 sfm_utils.py.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=4000,
        help="Maximum number of SIFT features to retain per image.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Resize images so their long edge is at most this size. Use 0 to disable.",
    )
    parser.add_argument(
        "--final-filter",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--optimization",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args

# main pipeline
def run(args: argparse.Namespace) -> None:
    # load week2 and week3 libraries
    week2 = load_week2_module(args.week2_dir)
    week3 = load_week3_module(args.week3_dir)

    # load images and extract features
    output_dir = week3.ensure_dir(args.output_dir)
    image_paths = week2.list_image_paths(args.image_dir, max_images=args.max_images)
    print(f"Loaded {len(image_paths)} images")
    features = week2.precompute_image_features(
        image_paths,
        max_features=args.max_features,
        max_image_size=args.max_image_size,
    )

    # initialize point cloud with two images that share the most number of matches
    init_i, init_j, count = select_initial_pair_by_matches(
        features=features,
        week2=week2,
        ratio=args.ratio,
        max_candidates=None,
        max_pair_gap=None,
    )
    print(f"Initial pair: ({init_i}, {init_j}), matches={count}")
    # get features of initial images
    features0 = features[init_i]
    features1 = features[init_j]
    # match features
    matches = week2.match_descriptors(features0.descriptors, features1.descriptors, ratio=args.ratio)
    # if the best pair has less than 8 matches, report error because this is a bad image set
    if len(matches) < 8:
        raise ValueError(f"Need at least 8 Lowe-filtered matches, got {len(matches)}")
    # get coordiante of matching features
    pts0, pts1 = week2.matched_keypoint_coords(features0.keypoints, features1.keypoints, matches)
    # estimate intrinsic parameter matrix
    K = week3.make_camera_matrix(
        features0.image.shape,
        focal_length_px=args.focal_length_px,
        principal_point=None if args.principal_point is None else tuple(args.principal_point),
    )
    # estimate essential matrix
    # essential_mask is those features used to get E
    E, essential_mask = week3.estimate_essential_matrix(
        pts0,
        pts1,
        K,
        threshold=args.ransac_threshold,
        confidence=args.confidence,
    )
    # estimate rotation and translation of second image
    # pose mask is those features used to estimate camera pose
    R, t, pose_mask = week3.recover_relative_pose(E, pts0, pts1, K, inlier_mask=essential_mask)
    pose_mask = pose_mask.reshape((pose_mask.shape[0])) == 1
    # get those features under pose mask
    pts0_pose = pts0[pose_mask]
    pts1_pose = pts1[pose_mask]
    # get those matches under pose mask
    pose_matches = [match for match, keep in zip(matches, pose_mask) if keep]
    # triangulate 3D points from features and mK, R, t
    points3d = week3.triangulate_points(pts0_pose, pts1_pose, K, R, t)
    # calculate reprojection error
    errors0 = week3.compute_reprojection_errors(points3d, pts0_pose, K, np.eye(3), np.zeros((3, 1)))
    errors1 = week3.compute_reprojection_errors(points3d, pts1_pose, K, R, t)
    # remove those 3D point with high reprojection error
    # keep is mask of those remained points
    keep = week3.filter_reconstructed_points(
        points3d,
        errors0,
        errors1,
        R,
        t,
        max_reprojection_error=args.max_reprojection_error,
    )
    # # calculate depth of points
    # depths0, depths1 = week3.compute_depths(points3d, R, t)
    # # keep those points in front of camera
    # positive_depth = (depths0 > 0) & (depths1 > 0)
    kept_points = points3d[keep]
    # extract color for 3D points
    kept_colours = week3.sample_point_colours(features0.image, pts0_pose[keep])
    # get those matches used to reconstruction
    kept_pose_matches = [match for match, keep_point in zip(pose_matches, keep) if keep_point]
    

# --------------------------------------------------------------------------------------------
    # # test first stage
    # week3.plot_patch_cloud_reconstruction(
    #     kept_points,
    #     features1.image,
    #     pts1_pose[keep],
    #     [
    #         ("Camera 1", np.eye(3), np.zeros((3, 1))),
    #         ("Camera 2", R, t),
    #     ],
    #     output_dir / "two_view_patch_cloud.png",
    # )
    # week3.write_ply(output_dir / "points3d.ply", kept_points, kept_colours)
# --------------------------------------------------------------------------------------------

    # introduce more images
    # first step is PnP pose estimation of cameras
    # register camera 1 and camera 2
    # camera pose is a dictionary to registered camera pose
    # eg camera_pose = {4: (R4, t4), 5: (R5, t5)}
    camera_pose = {}
    # first images is origin
    camera_pose[init_i] = (np.eye(3), np.zeros((3, 1)))
    camera_pose[init_j] = (R, t)
    # tracks store background information of a 3D point
    # eg tracks[6] = {0:12, 1:4} means point 6 is observed in camera 0 at keypoint 12 and in camera 1 at keypoint 4
    tracks = []
    # obs_to_point stores mapping a keypoint in a camera to a 3D point
    # eg obs_to_point = {(0,12):6, (1,4):6}
    obs_to_point = {}
    # store information from camera 1 and camera 2
    for point_id, match in enumerate(kept_pose_matches):
        kp0 = match.queryIdx
        kp1 = match.trainIdx
        track = {
            init_i: kp0,
            init_j: kp1,
        }
        tracks.append(track)
        obs_to_point[(init_i, kp0)] = point_id
        obs_to_point[(init_j, kp1)] = point_id
    # register camera 1 and camera 2
    registered_images = {init_i, init_j}
    # store those unregistered image
    unregistered_images = set(range(len(image_paths))) - registered_images  
    # there are three image states: registered, unregistered but analyzed, unregistered and not analyzed
    # unregistered_images stores unregistered and not analyzed images
    # registered_images stores registered_images images
    # unregistered but analyzed images do not appear in either of them

    # start registering more images
    # keep iterating until there is no image in unregistered_images
    while unregistered_images:
        # select the next image with the most total number of matches with registered images
        j, total_match = select_next_image_by_matches(
            registered_images=registered_images,
            unregistered_images=unregistered_images,
            features=features,
            week2=week2,
            ratio=args.ratio,
        )
        # if next image is not found, stop iterating
        if j is None:
            break
        # show information about next image
        print(f"Next image: {j}, total_matches={total_match}")
        # chunks is used to store relationship between analyzing image with all registered images
        # chunks is used later for filtering
        chunks = []
        # get pose of camera through PnP
        features_j = features[j]
        # collect relationship with all registered images
        for anchor_i in registered_images:
            # get macthes
            matches = week2.match_descriptors(
                features[anchor_i].descriptors, features_j.descriptors, ratio=args.ratio
            )
            # filter those 3D points and necessary information needed to do PnP pose estimation
            # make sure feature_i has corresponding 3D points and each point_j is used once
            # each feature_j can only map to one feature_i (otherwise this makes no sense)
            pts3d, pts2d, point_ids, kp_idx_j = build_pnp_from_match(
                anchor_i, j, matches, features, kept_points, obs_to_point
            )
            # make sure there is meaningful points
            if len(pts3d) > 0:
                chunks.append((pts3d, pts2d, point_ids, kp_idx_j))
        # if overall length of chunk is 0, this image is skipped
        if len(chunks) == 0:
            print(f"image {j} skipped: no 2D-3D correspondences")
            unregistered_images.remove(j)
            continue
        # merge all elements in trunks
        # make sure each feature_j is used once
        # if feature_j is mapped to multiple features in different images
        # there are two situations
        # 1. these features represent same 3D point -> keep one is enough
        # 2. these features represent different 3D points -> keep one (the feature_j might be meaningless) but we still have PnP inlier filter
        pnp_points3d, pnp_pts2d, pnp_point_ids, pnp_kp_indices_j = merge_pnp_chunks(chunks)
        # only do PnP if number of points is large enough
        if len(pnp_points3d) < 6:
            print(f"image {j} skipped: not enough PnP correspondences")
            continue
        # do PnP estimation to get rotation and translation
        # PnP inliers mean those points used to do PnP
        Rj, tj, pnp_inliers = week3.estimate_camera_pose_pnp(
            pnp_points3d,
            pnp_pts2d,
            K,
            threshold=6.0,
            confidence=args.confidence,
        )
        # is no points are used, skip this image
        if pnp_inliers is None:
            unregistered_images.remove(j)
            continue
        # calculate inlier ratio
        inlier_ratio = len(pnp_inliers) / len(pnp_points3d)
        # skip the image if inlier ratio is too low
        if inlier_ratio < 0.3:
            unregistered_images.remove(j)
            continue
        else:
            # if all tests are passed, register j
            registered_images.add(j)
            unregistered_images.remove(j)
            camera_pose[j] = (Rj, tj)
            # update information in tracks and obs_to_point
            for idx in pnp_inliers.reshape(-1):
                point_id = pnp_point_ids[idx]
                kp_j = pnp_kp_indices_j[idx]
                if (j, kp_j) not in obs_to_point:
                    tracks[point_id][j] = kp_j
                    obs_to_point[(j, kp_j)] = point_id
#---------------------------------------------------------------------
    # print(f"registering image {j}")
    # print(f"  anchor chunks: {len(chunks)}")
    # print(f"  merged PnP correspondences: {len(pnp_points3d)}")
    # print(f"  PnP inliers: {len(pnp_inliers.reshape(-1))}")
    # print(f"  registered cameras: {sorted(camera_pose.keys())}")
#---------------------------------------------------------------------
        # start to triangulate new points
        for i in sorted(camera_pose.keys()):
            if i == j:
                continue
            # get matches
            matches_ij = week2.match_descriptors(
                features[i].descriptors,
                features[j].descriptors,
                ratio=args.ratio,
            )
            # make sure both features do not have existing 3D points
            # avoid one feature mapped to multiple features
            # avoid multiple features mapped to one feature
            pts_i, pts_j, kp_indices_i, kp_indices_j = collect_new_point_matches(
                i, j, matches_ij, features, obs_to_point
            )
            # we need at least two matches
            if len(pts_i) < 2:
                continue
            # get rotation and translation matix
            Ri, ti = camera_pose[i]
            Rj, tj = camera_pose[j]
            # triangulate new points
            # this function is almost same as week3 function
            # however week3 function assumes image 1 is origin which is not the case here
            candidate_points3d = triangulate_points_new(
                pts_i,
                pts_j,
                K,
                Ri,
                ti,
                Rj,
                tj,
            )
            # calculate reprojection errors
            errors_i = week3.compute_reprojection_errors(candidate_points3d, pts_i, K, Ri, ti)
            errors_j = week3.compute_reprojection_errors(candidate_points3d, pts_j, K, Rj, tj)
            # filter points by checking reprojection error
            # this function is almost same as week3 function
            # however week3 function assumes image 1 is origin which is not the case here
            keep_new = filter_reconstructed_points_new(
                candidate_points3d,
                errors_i,
                errors_j,
                Ri,
                ti,
                Rj,
                tj,
                week3
            )
            # if no points are left, skip this image
            if not np.any(keep_new):
                continue
            # apply mask to key information
            new_points3d = candidate_points3d[keep_new]
            new_pts_j = pts_j[keep_new]
            new_kp_indices_i = [kp for kp, keep_flag in zip(kp_indices_i, keep_new) if keep_flag]
            new_kp_indices_j = [kp for kp, keep_flag in zip(kp_indices_j, keep_new) if keep_flag]
            new_colours = week3.sample_point_colours(features[j].image, new_pts_j)
            # add new points and colors to existing data
            kept_points, kept_colours = append_new_points(
                kept_points,
                kept_colours,
                tracks,
                obs_to_point,
                i,
                j,
                new_points3d,
                new_colours,
                new_kp_indices_i,
                new_kp_indices_j,
            )
        # # regular least square optimization
        # if args.optimization == 1 and (len(camera_pose) == 5 or len(camera_pose) == 10):
        #     all_camera_ids = sorted(camera_pose.keys())
        #     fixed_camera_id = all_camera_ids[0]
        #     camera_pose, kept_points, ba_stats = run_bundle_adjustment(
        #         camera_pose=camera_pose,
        #         points3d=kept_points,
        #         tracks=tracks,
        #         features=features,
        #         K=K,
        #         camera_ids=all_camera_ids,
        #         fixed_camera_id=fixed_camera_id,
        #         min_track_observations=2,
        #         robust_loss="huber",
        #         f_scale=2.0,
        #         max_nfev=10,
        #         verbose=1,
        #     )

    # if use open3d:
    # unset WAYLAND_DISPLAY
    # export DISPLAY=:0
    # export XDG_SESSION_TYPE=x11
    # export LIBGL_ALWAYS_SOFTWARE=1
    # sudo apt install x11-apps
 
    # # final global optimization
    # if args.optimization == 1:
    #     all_camera_ids = sorted(camera_pose.keys())
    #     fixed_camera_id = all_camera_ids[0]
    #     camera_pose, kept_points, ba_stats = run_bundle_adjustment(
    #         camera_pose=camera_pose,
    #         points3d=kept_points,
    #         tracks=tracks,
    #         features=features,
    #         K=K,
    #         camera_ids=all_camera_ids,
    #         fixed_camera_id=fixed_camera_id,
    #         min_track_observations=2,
    #         robust_loss="huber",
    #         f_scale=2.0,
    #         max_nfev=10,
    #         verbose=1,
    #     )

    # apply final filter
    # remove those noisy points
    if args.final_filter == 1:
        # three filters
        # number of tracks (eg a 3D point should be observed by at least 3 images)
        # reprojection error
        # spatial outrlier remover (remove those points far away from 3D cloud center)
        kept_points, kept_colours, tracks, obs_to_point, old_point_num = prune_bad_points(
            kept_points,
            kept_colours,
            tracks,
            obs_to_point,
            camera_pose,
            features,
            K,
            week3,
            min_track_length=2,
            max_mean_error=99.0,# 3.0
            max_max_error=99.0,# 5.0
            spatial_keep_percentile=75.0,
        )
    # convert camera pose to list
    camera_pose_list = [
        (f"Camera {idx}", R_i, t_i)
        for idx, (R_i, t_i) in sorted(camera_pose.items())
    ]
    # print results
    print(f"{len(image_paths)} images are loaded")
    print(f"{len(camera_pose_list)} images are registered")
    if args.final_filter == 1:
        print(f"{old_point_num} points are registered before final filtering")
        print(f"{len(kept_points)} points are registered after final filtering")
    else:
        print(f"{len(kept_points)} points are registered")
    # store key results
    cam_names = [name for name, R, t in camera_pose_list]
    cam_Rs    = np.stack([R for _, R, _ in camera_pose_list])   # shape (N, 3, 3)
    cam_ts    = np.stack([t for _, _, t in camera_pose_list])   # shape (N, 3, 1)
    np.savez(
        output_dir / "final_scene.npz",
        points=kept_points,          # (M, 3)
        colours=kept_colours,        # (M, 3)
        cam_Rs=cam_Rs,               # (N, 3, 3)
        cam_ts=cam_ts,               # (N, 3, 1)
        cam_names=cam_names,         # list of strings
    )
    print(f"Scene saved to {output_dir / 'final_scene.npz'}")
    # visualization based on open3d
    visualize_open3d_scene(
        kept_points,
        kept_colours,
        camera_pose_list,
    )

# use plotly
#--------------------------------------------------------------------
    # plotly_multi_view_scene(
    #     kept_points,
    #     kept_colours,
    #     camera_pose_list,
    #     output_dir / "multi_view_scene.html",
    # ) 
#--------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    run(args)

if __name__ == "__main__":
    raise SystemExit(main())



