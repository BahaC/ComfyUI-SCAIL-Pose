import os
import torch
from tqdm import tqdm
import numpy as np
import folder_paths
import cv2
import logging
import copy
import datetime
script_directory = os.path.dirname(os.path.abspath(__file__))

from comfy import model_management as mm
from comfy.utils import ProgressBar
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

folder_paths.add_model_folder_path("detection", os.path.join(folder_paths.models_dir, "detection"))

from .vitpose_utils.utils import bbox_from_detector, crop, load_pose_metas_from_kp2ds_seq, aaposemeta_to_dwpose_scail

def convert_openpose_to_target_format(frames, max_people=2):
    NUM_BODY = 18
    NUM_FACE = 70
    NUM_HAND = 21

    results = []
    for frame in frames:
        canvas_width = frame['canvas_width']
        canvas_height = frame['canvas_height']
        people = frame['people'][:max_people]

        bodies = []
        hands = []
        faces = []
        body_scores = []
        hand_scores = []
        face_scores = []

        for person in people:
            pose_raw = person.get('pose_keypoints_2d') or []
            if len(pose_raw) != NUM_BODY * 3:
                continue

            pose = np.array(pose_raw).reshape(-1, 3)
            pose_xy = np.stack([pose[:, 0] / canvas_width, pose[:, 1] / canvas_height], axis=1)
            bodies.append(pose_xy)
            body_scores.append(pose[:, 2])

            face_raw = person.get('face_keypoints_2d') or []
            if len(face_raw) == NUM_FACE * 3:
                face = np.array(face_raw).reshape(-1, 3)
                face_xy = np.stack([face[:, 0] / canvas_width, face[:, 1] / canvas_height], axis=1)
                faces.append(face_xy)
                face_scores.append(face[:, 2])

            hand_left_raw = person.get('hand_left_keypoints_2d') or []
            hand_right_raw = person.get('hand_right_keypoints_2d') or []
            if len(hand_left_raw) == NUM_HAND * 3:
                hand_left = np.array(hand_left_raw).reshape(-1, 3)
                hand_left_xy = np.stack([hand_left[:, 0] / canvas_width, hand_left[:, 1] / canvas_height], axis=1)
                hands.append(hand_left_xy)
                hand_scores.append(hand_left[:, 2])
            if len(hand_right_raw) == NUM_HAND * 3:
                hand_right = np.array(hand_right_raw).reshape(-1, 3)
                hand_right_xy = np.stack([hand_right[:, 0] / canvas_width, hand_right[:, 1] / canvas_height], axis=1)
                hands.append(hand_right_xy)
                hand_scores.append(hand_right[:, 2])

        result = {
            'bodies': {
                'candidate': np.array(bodies, dtype=np.float32),
                'subset': np.array([np.arange(NUM_BODY) for _ in bodies], dtype=np.float32) if bodies else np.array([])
            },
            'hands': np.array(hands, dtype=np.float32),
            'faces': np.array(faces, dtype=np.float32),
            'body_score': np.array(body_scores, dtype=np.float32),
            'hand_score': np.array(hand_scores, dtype=np.float32),
            'face_score': np.array(face_scores, dtype=np.float32)
        }
        results.append(result)
    return results

def scale_faces(poses, pose_2d_ref):
    # Input: two lists of dict, poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2
    # Scale the facial keypoints in poses according to the center point of the face
    # That is: calculate the distance from the center point (idx: 30) to other facial keypoints in ref,
    # and the same for poses, then get scale_n as the ratio
    # Clamp scale_n to the range 0.8-1.5, then apply it to poses
    # Note: poses are modified in place

    ref = pose_2d_ref[0]
    pose_0 = poses[0]

    face_0 = pose_0['faces']  # shape: (1, 68, 2)
    face_ref = ref['faces']

    # Extract numpy arrays
    face_0 = np.array(face_0[0])      # (68, 2)
    face_ref = np.array(face_ref[0])

    # Center point (nose tip or face center)
    center_idx = 30
    center_0 = face_0[center_idx]
    center_ref = face_ref[center_idx]

    # Calculate distance to center point
    dist = np.linalg.norm(face_0 - center_0, axis=1)
    dist_ref = np.linalg.norm(face_ref - center_ref, axis=1)

    # Avoid the 0 distance of the center point itself
    dist = np.delete(dist, center_idx)
    dist_ref = np.delete(dist_ref, center_idx)

    mean_dist = np.mean(dist)
    mean_dist_ref = np.mean(dist_ref)

    if mean_dist < 1e-6:
        scale_n = 1.0
    else:
        scale_n = mean_dist_ref / mean_dist

    # Clamp to [0.8, 1.5]
    scale_n = np.clip(scale_n, 0.8, 1.5)

    for i, pose in enumerate(poses):
        face = pose['faces']
        # Extract numpy array
        face = np.array(face[0])      # (68, 2)
        center = face[center_idx]
        scaled_face = (face - center) * scale_n + center
        poses[i]['faces'][0] = scaled_face

        body = pose['bodies']
        candidate = body['candidate']
        candidate_np = np.array(candidate[0])   # (14, 2)
        body_center = candidate_np[0]
        scaled_candidate = (candidate_np - body_center) * scale_n + body_center
        poses[i]['bodies']['candidate'][0] = scaled_candidate

    # In-place modification
    pose['faces'][0] = scaled_face

    return scale_n

def merge_dwpose_results(person_dwposes):
    """Merge multiple single-person DWPose dicts into one multi-person dict."""
    if len(person_dwposes) == 1:
        return person_dwposes[0]
    return {
        "bodies": {
            "candidate": np.concatenate([p["bodies"]["candidate"] for p in person_dwposes], axis=0),
            "subset": np.concatenate([p["bodies"]["subset"] for p in person_dwposes], axis=0),
        },
        "hands": np.concatenate([p["hands"] for p in person_dwposes], axis=0),
        "faces": np.concatenate([p["faces"] for p in person_dwposes], axis=0),
        "body_score": np.concatenate([p["body_score"] for p in person_dwposes], axis=0),
        "hand_score": np.concatenate([p["hand_score"] for p in person_dwposes], axis=0),
        "face_score": np.concatenate([p["face_score"] for p in person_dwposes], axis=0),
    }

class PoseDetectionVitPoseToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vitpose_model": ("POSEMODEL",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "ViTPose to DWPose format pose detection node."

    def process(self, vitpose_model, images):

        detector = vitpose_model["yolo"]
        pose_model = vitpose_model["vitpose"]
        B, H, W, C = images.shape

        shape = np.array([H, W])[None]
        images_np = images.numpy()

        IMG_NORM_MEAN = np.array([0.485, 0.456, 0.406])
        IMG_NORM_STD = np.array([0.229, 0.224, 0.225])
        input_resolution=(256, 192)
        rescale = 1.25

        detector.reinit()
        pose_model.reinit()

        comfy_pbar = ProgressBar(B*2)
        progress = 0

        bboxes_per_frame = []
        for img in tqdm(images_np, total=len(images_np), desc="Detecting bboxes"):
            detections = detector(
                cv2.resize(img, (640, 640)).transpose(2, 0, 1)[None],
                shape,
                single_person=False
                )[0]
            frame_bboxes = []
            for det in detections:
                bbox = det["bbox"]
                if bbox is not None and bbox[-1] > 0 and (bbox[2] - bbox[0]) >= 10 and (bbox[3] - bbox[1]) >= 10:
                    frame_bboxes.append(bbox)
            if not frame_bboxes:
                frame_bboxes = [np.array([0, 0, W, H])]
            bboxes_per_frame.append(frame_bboxes)
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        detector.cleanup()

        dwposes = []
        for img, frame_bboxes in tqdm(zip(images_np, bboxes_per_frame), total=len(images_np), desc="Extracting keypoints"):
            person_dwposes = []
            for bbox in frame_bboxes:
                center, scale = bbox_from_detector(bbox, input_resolution, rescale=rescale)
                cropped = crop(img, center, scale, (input_resolution[0], input_resolution[1]))[0]

                img_norm = (cropped - IMG_NORM_MEAN) / IMG_NORM_STD
                img_norm = img_norm.transpose(2, 0, 1).astype(np.float32)

                keypoints = pose_model(img_norm[None], np.array(center)[None], np.array(scale)[None])
                meta = load_pose_metas_from_kp2ds_seq(keypoints, width=W, height=H)[0]
                person_dwposes.append(aaposemeta_to_dwpose_scail(meta))

            dwposes.append(merge_dwpose_results(person_dwposes))
            progress += 1
            if progress % 10 == 0:
                comfy_pbar.update_absolute(progress)

        pose_model.cleanup()

        swap_hands = True
        out_dict = {"poses": dwposes, "swap_hands": swap_hands}
        return out_dict,


class ConvertOpenPoseKeypointsToDWPose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints": ("POSE_KEYPOINT",),
                "max_people": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1, "tooltip": "Maximum number of people to process per frame"}),
            },
        }

    RETURN_TYPES = ("DWPOSES",)
    RETURN_NAMES = ("dw_poses",)
    FUNCTION = "process"
    CATEGORY = "WanAnimatePreprocess"
    DESCRIPTION = "Convert OpenPose format keypoints to DWPose format."

    def process(self, keypoints, max_people=2):
        swap_hands = False
        out_dict = {"poses": convert_openpose_to_target_format(keypoints, max_people=max_people), "swap_hands": swap_hands}
        return out_dict,


def filter_to_single_person(pose_input, dw_pose_input, intrinsic_matrix, height, width):
    """Filter multi-person NLF and DWPose inputs to the main character only.

    Main character = largest projected 2D bounding box in first valid frame, tracked by pelvis proximity.
    DWPose person is matched by projecting the NLF main person's head joint to 2D.
    """
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]

    main_idx = 0
    for frame_poses in pose_input:
        if frame_poses.shape[0] == 0:
            continue
        max_area = -1
        for p_idx in range(frame_poses.shape[0]):
            person = frame_poses[p_idx]
            person_np = person.cpu().numpy() if isinstance(person, torch.Tensor) else person
            if np.sum(np.abs(person_np)) < 0.01:
                continue
            valid = person_np[:, 2] > 0.01
            if not np.any(valid):
                continue
            pts = person_np[valid]
            u = (fx * pts[:, 0] / pts[:, 2] + cx) / width
            v = (fy * pts[:, 1] / pts[:, 2] + cy) / height
            area = (np.max(u) - np.min(u)) * (np.max(v) - np.min(v))
            if area > max_area:
                max_area = area
                main_idx = p_idx
        break

    tracked_nlf_indices = []
    prev_pelvis = None
    for frame_poses in pose_input:
        if frame_poses.shape[0] == 0:
            tracked_nlf_indices.append(0)
            continue
        if prev_pelvis is None:
            tracked_idx = main_idx if main_idx < frame_poses.shape[0] else 0
        else:
            min_dist = float('inf')
            tracked_idx = 0
            for p_idx in range(frame_poses.shape[0]):
                pelvis = frame_poses[p_idx][0]
                pelvis_np = pelvis.cpu().numpy() if isinstance(pelvis, torch.Tensor) else pelvis
                dist = np.linalg.norm(pelvis_np - prev_pelvis)
                if dist < min_dist:
                    min_dist = dist
                    tracked_idx = p_idx
        tracked_nlf_indices.append(tracked_idx)
        pelvis = frame_poses[tracked_idx][0]
        prev_pelvis = pelvis.cpu().numpy() if isinstance(pelvis, torch.Tensor) else pelvis

    filtered_pose_input = []
    for frame_idx, frame_poses in enumerate(pose_input):
        t_idx = tracked_nlf_indices[frame_idx]
        if frame_poses.shape[0] > 0 and t_idx < frame_poses.shape[0]:
            filtered_pose_input.append(frame_poses[t_idx:t_idx+1])
        elif frame_poses.shape[0] > 0:
            filtered_pose_input.append(frame_poses[0:1])
        else:
            filtered_pose_input.append(frame_poses)

    # Maximum normalized 0..1 distance between the NLF main person's
    # projected head and the closest DWPose person we're willing to accept.
    # 0.15 (~15% of image dimension) is much larger than typical
    # head-to-body-center / head-to-nose offsets for the same person, so
    # this is a "different person" guard rather than a tight match. When
    # exceeded, DWPose for that frame is dropped entirely so the wrong
    # person's face/hands don't get drawn onto the NLF skeleton (see the
    # frame-97 background-person leak case).
    DW_MATCH_THRESHOLD = 0.15

    if dw_pose_input is not None:
        for frame_idx, frame_dw in enumerate(dw_pose_input):
            num_dw_people = frame_dw['bodies']['candidate'].shape[0]
            if num_dw_people == 0:
                continue

            nlf_frame = filtered_pose_input[frame_idx]
            best_dw_idx = 0
            min_dist = float('inf')
            have_nlf_head = False

            if nlf_frame.shape[0] > 0:
                head = nlf_frame[0][15]  # NLF joint 15 = head/nose
                head_np = head.cpu().numpy() if isinstance(head, torch.Tensor) else head
                if np.sum(np.abs(head_np)) > 0.01 and head_np[2] > 0.01:
                    u = (fx * head_np[0] / head_np[2] + cx) / width
                    v = (fy * head_np[1] / head_np[2] + cy) / height
                    nlf_head_2d = np.array([u, v])
                    have_nlf_head = True

                    for dw_p_idx in range(num_dw_people):
                        dw_body = frame_dw['bodies']['candidate'][dw_p_idx]
                        dw_subset = frame_dw['bodies']['subset'][dw_p_idx]
                        # Prefer nose-to-nose (DWPose joint 0 in COCO 18) when
                        # the nose is visible; fall back to body center over
                        # visible keypoints when not.
                        if dw_subset[0] != -1.0 and np.any(dw_body[0] != 0):
                            dw_ref = dw_body[0]
                        else:
                            valid = np.any(dw_body != 0, axis=1)
                            if np.any(valid):
                                dw_ref = np.mean(dw_body[valid], axis=0)
                            else:
                                dw_ref = np.mean(dw_body, axis=0)
                        dist = float(np.linalg.norm(nlf_head_2d - dw_ref))
                        if dist < min_dist:
                            min_dist = dist
                            best_dw_idx = dw_p_idx

            # Drop this frame's DWPose entirely when even the closest
            # match is implausibly far from the NLF-tracked head: typically
            # means YOLO/DWPose only saw a different person (e.g. someone in
            # the background) at that frame, so keeping their face/hands
            # would draw the wrong person's head on top of the right person's
            # body. shift_dwpose_according_to_nlf already handles count
            # mismatches (NLF: 1, DW: 0) by skipping the per-frame shift, and
            # draw_pose_to_canvas_np then has no 2D detail to overlay -- the
            # NLF body cylinders (including Neck->Nose) still render.
            if have_nlf_head and min_dist > DW_MATCH_THRESHOLD:
                logging.warning(
                    "filter_to_single_person: frame %d: best DWPose match is "
                    "%.3f normalized units from NLF head (>%.2f); dropping "
                    "DWPose for this frame to prevent wrong-person face/hands.",
                    frame_idx, min_dist, DW_MATCH_THRESHOLD,
                )
                frame_dw['bodies']['candidate'] = frame_dw['bodies']['candidate'][0:0]
                frame_dw['bodies']['subset'] = frame_dw['bodies']['subset'][0:0]
                frame_dw['hands'] = frame_dw['hands'][0:0]
                frame_dw['faces'] = frame_dw['faces'][0:0]
                if 'body_score' in frame_dw:
                    frame_dw['body_score'] = frame_dw['body_score'][0:0]
                if 'hand_score' in frame_dw:
                    frame_dw['hand_score'] = frame_dw['hand_score'][0:0]
                if 'face_score' in frame_dw:
                    frame_dw['face_score'] = frame_dw['face_score'][0:0]
                continue

            p = best_dw_idx
            frame_dw['bodies']['candidate'] = frame_dw['bodies']['candidate'][p:p+1]
            frame_dw['bodies']['subset'] = frame_dw['bodies']['subset'][p:p+1]
            frame_dw['hands'] = frame_dw['hands'][2*p:2*p+2]
            frame_dw['faces'] = frame_dw['faces'][p:p+1]
            if 'body_score' in frame_dw:
                frame_dw['body_score'] = frame_dw['body_score'][p:p+1]
            if 'hand_score' in frame_dw:
                frame_dw['hand_score'] = frame_dw['hand_score'][2*p:2*p+2]
            if 'face_score' in frame_dw:
                frame_dw['face_score'] = frame_dw['face_score'][p:p+1]

    return filtered_pose_input, dw_pose_input


# NLF SMPL joint indices that feed the COCO arm bones via
# `process_data_to_COCO_format`: 16/17 = R/L shoulder, 18/19 = R/L elbow,
# 20/21 = R/L wrist. The cylinder builder in nlf_render skips any bone whose
# endpoint is all-zero, so when NLF briefly loses one of these joints the whole
# arm vanishes from the render. The two helpers below patch those gaps before
# rendering.
_NLF_ARM_JOINT_INDICES = (16, 17, 18, 19, 20, 21)


def _joint_to_np(joint):
    """Return a numpy view of a single (3,) NLF joint, regardless of source type."""
    if isinstance(joint, torch.Tensor):
        return joint.detach().cpu().numpy()
    return np.asarray(joint)


def _set_joint_inplace(pose_frame, joint_idx, xyz):
    """Write `xyz` (length-3 array-like) into `pose_frame[0][joint_idx]` in place.

    `pose_frame` is `pose_input[frame_idx]` of shape (1, 24, 3). It can be a
    torch tensor or a numpy array; both support indexed assignment, but tensors
    need a tensor RHS on the right device/dtype.
    """
    if isinstance(pose_frame, torch.Tensor):
        rhs = torch.as_tensor(xyz, dtype=pose_frame.dtype, device=pose_frame.device)
        pose_frame[0][joint_idx] = rhs
    else:
        pose_frame[0][joint_idx] = np.asarray(xyz, dtype=pose_frame.dtype)


def _interpolate_arm_joints_temporal(pose_input):
    """Linearly interpolate NLF arm joints across frames where NLF dropped them.

    Operates per joint in `_NLF_ARM_JOINT_INDICES`. A joint is considered
    "missing" when its (X, Y, Z) is all-zero (NLF's convention for a dropped
    joint, matching the same `np.sum(np.abs(...)) < 0.01` test used elsewhere
    in this file). For each missing frame the function finds the nearest valid
    frame before and after; if both exist, it linearly interpolates the 3D
    coords by frame distance; if only one side exists, that side's value is
    carried over. Joints that are missing across the entire clip are left for
    `_backfill_arm_joints_from_dwpose` to handle.

    Assumes `pose_input` is the post-`filter_to_single_person` list where each
    frame's shape is (1, 24, 3) (single tracked person). Mutates in place.
    """
    n_frames = len(pose_input)
    if n_frames == 0:
        return

    for joint_idx in _NLF_ARM_JOINT_INDICES:
        coords = [None] * n_frames
        for f_idx in range(n_frames):
            frame = pose_input[f_idx]
            if frame is None or frame.shape[0] == 0:
                continue
            jnp = _joint_to_np(frame[0][joint_idx])
            if np.sum(np.abs(jnp)) > 0.01:
                coords[f_idx] = jnp.astype(np.float32, copy=True)

        if not any(c is not None for c in coords):
            continue

        for f_idx in range(n_frames):
            if coords[f_idx] is not None:
                continue
            frame = pose_input[f_idx]
            if frame is None or frame.shape[0] == 0:
                continue

            before_idx = None
            for i in range(f_idx - 1, -1, -1):
                if coords[i] is not None:
                    before_idx = i
                    break
            after_idx = None
            for i in range(f_idx + 1, n_frames):
                if coords[i] is not None:
                    after_idx = i
                    break

            if before_idx is not None and after_idx is not None:
                span = float(after_idx - before_idx)
                t = float(f_idx - before_idx) / span
                interpolated = coords[before_idx] * (1.0 - t) + coords[after_idx] * t
            elif before_idx is not None:
                interpolated = coords[before_idx]
            elif after_idx is not None:
                interpolated = coords[after_idx]
            else:
                continue

            _set_joint_inplace(frame, joint_idx, interpolated)


# Mapping from NLF arm joint index to (DWPose COCO-18 candidate index,
# NLF parent joint index used for the back-projection depth). The parent
# choice walks the kinematic chain neck -> shoulder -> elbow -> wrist so
# we read Z from a joint that's almost always closer to the camera than
# the missing one and is least likely to itself be dropped.
_NLF_TO_DW_ARM_MAP = {
    16: (5, 12),  # R shoulder, parent: neck
    17: (2, 12),  # L shoulder, parent: neck
    18: (6, 16),  # R elbow,    parent: R shoulder
    19: (3, 17),  # L elbow,    parent: L shoulder
    20: (7, 18),  # R wrist,    parent: R elbow
    21: (4, 19),  # L wrist,    parent: L elbow
}


def _backfill_arm_joints_from_dwpose(pose_input, dw_pose_input, intrinsic_matrix, height, width):
    """Back-project DWPose 2D arm keypoints to 3D for joints NLF still hasn't filled.

    Runs AFTER `_interpolate_arm_joints_temporal`, so it only ever touches
    joints that are missing for an entire neighbourhood (no valid frame to
    interpolate from) but where DWPose still has the corresponding 2D body
    keypoint. The depth (Z) is borrowed from the parent NLF joint along the
    kinematic chain, with neck and then pelvis as further fallbacks; the
    pinhole back-projection then places the joint at the correct 2D position
    so the cylinder Builder draws the upper-arm / forearm bone normally.

    Skips silently when no DWPose data exists for the frame, when the DWPose
    keypoint is flagged missing (`subset[dw_idx] == -1.0` or zero coords), or
    when no usable depth source is available.
    """
    if dw_pose_input is None:
        return

    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    n_frames = len(pose_input)
    for f_idx in range(n_frames):
        frame = pose_input[f_idx]
        if frame is None or frame.shape[0] == 0:
            continue
        if f_idx >= len(dw_pose_input):
            break
        frame_dw = dw_pose_input[f_idx]
        try:
            dw_cand = np.asarray(frame_dw['bodies']['candidate'], dtype=np.float32)
            dw_subset = np.asarray(frame_dw['bodies']['subset'], dtype=np.float32)
        except (KeyError, TypeError):
            continue
        if dw_cand.ndim != 3 or dw_cand.shape[0] == 0:
            continue
        dw_body = dw_cand[0]
        dw_sub = dw_subset[0]

        for nlf_idx, (dw_idx, parent_idx) in _NLF_TO_DW_ARM_MAP.items():
            jnp = _joint_to_np(frame[0][nlf_idx])
            if np.sum(np.abs(jnp)) > 0.01:
                continue

            if dw_idx >= dw_body.shape[0] or dw_idx >= dw_sub.shape[0]:
                continue
            if dw_sub[dw_idx] == -1.0:
                continue
            u_norm, v_norm = float(dw_body[dw_idx, 0]), float(dw_body[dw_idx, 1])
            if u_norm == 0.0 and v_norm == 0.0:
                continue

            Z = 0.0
            for cand_parent in (parent_idx, 12, 0):
                parent_np = _joint_to_np(frame[0][cand_parent])
                if np.sum(np.abs(parent_np)) > 0.01 and parent_np[2] > 0.01:
                    Z = float(parent_np[2])
                    break
            if Z <= 0.0:
                continue

            u_px = u_norm * float(width)
            v_px = v_norm * float(height)
            X = (u_px - cx) * Z / fx
            Y = (v_px - cy) * Z / fy
            _set_joint_inplace(frame, nlf_idx, (X, Y, Z))


class RenderNLFPoses:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "width": ("INT", {"default": 512}),
            "height": ("INT", {"default": 512}),
            },
            "optional": {
                "dw_poses": ("DWPOSES", {"default": None, "tooltip": "Optional DW pose model for 2D drawing"}),
                "ref_dw_pose": ("DWPOSES", {"default": None, "tooltip": "Optional reference DW pose model for alignment"}),
                "draw_face": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw face keypoints"}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw hand keypoints"}),
                "render_device": (["gpu", "cpu", "opengl", "cuda", "vulkan", "metal"], {"default": "gpu", "tooltip": "Taichi device to use for rendering"}),
                "scale_hands": ("BOOLEAN", {"default": True, "tooltip": "Whether to scale hand keypoints when aligning DW poses"}),
                "render_backend": (["taichi", "torch"], {"default": "taichi", "tooltip": "Rendering backend to use"}),
                "single_person": ("BOOLEAN", {"default": False, "tooltip": "When True, select the main character from NLF (largest 3D body in first frame) and filter both NLF and DWPose to that one person."}),
                "fit_in_canvas": ("BOOLEAN", {"default": True, "tooltip": "Auto-scale the projected pose to fit inside the canvas. Prevents head/feet clipping when align3d's solver picks a camera that zooms in past the canvas edges. Uniform scale-down with re-center; no-op when the pose already fits."}),
                "fit_margin_px": ("INT", {"default": 12, "min": 0, "max": 128, "step": 1, "tooltip": "Minimum pixel margin between the projected pose AABB and every canvas edge after fitting. Default 12 covers the typical ~10-11 px cylinder radius; widen if heads/hands still touch the edge."}),
            }
    }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "predict"
    CATEGORY = "WanVideoWrapper"

    def predict(self, nlf_poses, width, height, dw_poses=None, ref_dw_pose=None, draw_face=True, draw_hands=True, render_device="gpu", scale_hands=True, render_backend="taichi", single_person=False, fit_in_canvas=True, fit_margin_px=12):

        from .NLFPoseExtract.nlf_render import render_nlf_as_images, render_multi_nlf_as_images, shift_dwpose_according_to_nlf, process_data_to_COCO_format, intrinsic_matrix_from_field_of_view
        from .NLFPoseExtract.align3d import solve_new_camera_params_central, solve_new_camera_params_down, fit_intrinsic_to_canvas
        if render_backend == "taichi":
            try:
                import taichi as ti
                device_map = {
                    "cpu": ti.cpu,
                    "gpu": ti.gpu,
                    "opengl": ti.opengl,
                    "cuda": ti.cuda,
                    "vulkan": ti.vulkan,
                    "metal": ti.metal,
                }
                ti.init(arch=device_map.get(render_device.lower()))
            except:
                logging.warning("Taichi selected but not installed. Falling back to torch rendering.")
                render_backend = "torch"

        if isinstance(nlf_poses, dict):
            pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
        else:
            pose_input = nlf_poses

        dw_pose_input = copy.deepcopy(dw_poses["poses"]) if dw_poses is not None else None
        swap_hands = dw_poses.get("swap_hands", False) if dw_poses is not None else False

        ori_camera_pose = intrinsic_matrix_from_field_of_view([height, width])
        ori_focal = ori_camera_pose[0, 0]

        if single_person:
            pose_input, dw_pose_input = filter_to_single_person(pose_input, dw_pose_input, ori_camera_pose, height, width)
            # Patch the runs of frames where NLF drops elbow/wrist joints so
            # the cylinder builder can still draw the arm bones. Temporal pass
            # first (nearest-neighbour interpolation of arm joints across
            # frames), then DWPose 2D back-projection for any joints still
            # missing afterwards. See _interpolate_arm_joints_temporal and
            # _backfill_arm_joints_from_dwpose for details.
            _interpolate_arm_joints_temporal(pose_input)
            _backfill_arm_joints_from_dwpose(pose_input, dw_pose_input, ori_camera_pose, height, width)

        num_people = dw_pose_input[0]['bodies']['candidate'].shape[0] if dw_pose_input is not None else 0

        if dw_poses is not None and ref_dw_pose is not None and num_people == 1:
            ref_dw_pose_input = copy.deepcopy(ref_dw_pose["poses"])

            # Find the first valid pose
            pose_3d_first_driving_frame = None
            for pose in pose_input:
                if pose.shape[0] == 0:
                    continue
                candidate = pose[0].cpu().numpy() if isinstance(pose[0], torch.Tensor) else np.asarray(pose[0])
                if np.any(candidate):
                    pose_3d_first_driving_frame = candidate
                    break
            if pose_3d_first_driving_frame is None:
                raise ValueError("No valid pose found in pose_input.")

            pose_3d_coco_first_driving_frame = process_data_to_COCO_format(pose_3d_first_driving_frame)
            poses_2d_ref = ref_dw_pose_input[0]['bodies']['candidate'][0][:14]
            poses_2d_ref[:, 0] = poses_2d_ref[:, 0] * width
            poses_2d_ref[:, 1] = poses_2d_ref[:, 1] * height

            poses_2d_subset = ref_dw_pose_input[0]['bodies']['subset'][0][:14]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[:14]

            valid_indices, valid_upper_indices, valid_lower_indices = [], [], []
            upper_body_indices = [0, 2, 3, 5, 6]
            lower_body_indices = [9, 10, 12, 13]

            for i in range(len(poses_2d_subset)):
                if poses_2d_subset[i] != -1.0 and np.sum(pose_3d_coco_first_driving_frame[i]) != 0:
                    if i in upper_body_indices:
                        valid_upper_indices.append(i)
                    if i in lower_body_indices:
                        valid_lower_indices.append(i)

            valid_indices = [1] + valid_lower_indices if len(valid_upper_indices) < 4 else [1] + valid_lower_indices + valid_upper_indices # align body or only lower body

            pose_2d_ref = poses_2d_ref[valid_indices]
            pose_3d_coco_first_driving_frame = pose_3d_coco_first_driving_frame[valid_indices]

            if len(valid_lower_indices) >= 4:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_down(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)
            else:
                new_camera_intrinsics, scale_m, scale_s = solve_new_camera_params_central(pose_3d_coco_first_driving_frame, ori_focal, [height, width], pose_2d_ref)

            scale_face = scale_faces(list(dw_pose_input), list(ref_dw_pose_input))   # poses[0]['faces'].shape: 1, 68, 2  , poses_ref[0]['faces'].shape: 1, 68, 2

            logging.info(f"Scale - m: {scale_m}, face: {scale_face}")

            if fit_in_canvas:
                new_camera_intrinsics = fit_intrinsic_to_canvas(
                    new_camera_intrinsics, pose_input, height, width, fit_margin_px
                )

            shift_dwpose_according_to_nlf(pose_input, dw_pose_input, ori_camera_pose, new_camera_intrinsics, height, width, swap_hands=swap_hands, scale_hands=scale_hands, scale_x=scale_m, scale_y=scale_m*scale_s)

            intrinsic_matrix = new_camera_intrinsics
        else:
            intrinsic_matrix = ori_camera_pose

        if pose_input[0].shape[0] > 1:
            frames_np = render_multi_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)
        else:
            frames_np = render_nlf_as_images(pose_input, dw_pose_input, height, width, len(pose_input), intrinsic_matrix=intrinsic_matrix, draw_face=draw_face, draw_hands=draw_hands, render_backend = render_backend)

        frames_tensor = torch.from_numpy(np.stack(frames_np, axis=0)).contiguous() / 255.0
        frames_tensor, mask = frames_tensor[..., :3], frames_tensor[..., -1] > 0.5

        return (frames_tensor.cpu().float(), mask.cpu().float())

class SaveNLFPosesAs3D:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "nlf_poses": ("NLFPRED", {"tooltip": "Input poses for the model"}),
            "filename_prefix": ("STRING", {"default": "nlf_pose_3d"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 300.0, "step": 0.1, "tooltip": "Frames per second for the output animation"}),
            "cylinder_radius": ("FLOAT", {"default": 21.5, "tooltip": "Radius of the cylinders representing bones"}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_3d"
    CATEGORY = "WanVideoWrapper"

    def save_3d(self, nlf_poses, filename_prefix, fps, cylinder_radius):
        from .NLFPoseExtract.nlf_render import get_cylinder_specs_list_from_poses
        from .render_3d.export_utils import save_cylinder_specs_as_glb_animation
        try:
            if isinstance(nlf_poses, dict):
                pose_input = nlf_poses['joints3d_nonparam'][0] if 'joints3d_nonparam' in nlf_poses else nlf_poses
            else:
                pose_input = nlf_poses

            cylinder_specs_list = get_cylinder_specs_list_from_poses(pose_input, include_missing=True)
            logging.info(f"Generated {len(cylinder_specs_list)} frames of cylinder specs")

            output_dir = folder_paths.get_output_directory()
            full_output_folder = os.path.join(output_dir, filename_prefix)
            if not os.path.exists(full_output_folder):
                os.makedirs(full_output_folder)

            filename = f"{filename_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.glb"
            filepath = os.path.join(full_output_folder, filename)

            logging.info(f"Saving as GLB animation to {full_output_folder}")
            logging.info(f"Starting GLB animation export. Frames: {len(cylinder_specs_list)}")
            save_cylinder_specs_as_glb_animation(cylinder_specs_list, filepath, fps=fps, radius=cylinder_radius)
            logging.info(f"Saved GLB: {filepath}")
        except Exception as e:
            logging.error(f"Error in SaveNLFPosesAs3D: {e}")
            raise e

        return (filepath,)

NODE_CLASS_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": PoseDetectionVitPoseToDWPose,
    "RenderNLFPoses": RenderNLFPoses,
    "ConvertOpenPoseKeypointsToDWPose": ConvertOpenPoseKeypointsToDWPose,
    "SaveNLFPosesAs3D": SaveNLFPosesAs3D,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PoseDetectionVitPoseToDWPose": "Pose Detection VitPose to DWPose",
    "RenderNLFPoses": "Render NLF Poses",
    "ConvertOpenPoseKeypointsToDWPose": "Convert OpenPose Keypoints to DWPose",
    "SaveNLFPosesAs3D": "Save NLF Poses as 3D Animation",
}
