"""Multi-person pose-canvas duplication for Lisa S3 / SCAIL pet-dance pipeline.

This module adds two ComfyUI custom nodes that together implement
"Approach A+" from `diagnosis-v2.md` §M2: when the driving video has a
single dancer but the reference image contains multiple people, replicate
the rendered pose canvas onto every detected reference-image bbox at a
similar depth so `WanVideoAddSCAILPoseEmbeds` sees N skeletons in the
right places.

Nodes:

- ``LisaBBoxesFromDWPose``: derives per-person bounding boxes from a
  ``DWPOSES`` payload (the same dict produced by
  ``PoseDetectionVitPoseToDWPose.process``). Keypoints in DWPose are
  normalized 0..1 by the canvas they were detected on; the canvas size
  is not stored in the dict, so this node accepts the canvas size either
  as widget integers or by inferring from an optional reference IMAGE
  input. The output ``BBOXES`` is a small dict bundling the boxes with
  the canvas size so the consumer never has to guess.

- ``LisaPoseDuplicateByBBoxes``: takes the rendered pose-canvas
  sequence from ``RenderNLFPoses`` together with reference-image
  bboxes and composites one rescaled clone of the dancer per bbox.
  Scaling is **anchor-first**: the most-centered reference bbox (max
  ``area * x_centeredness * y_centeredness``) is the anchor and its
  clone is drawn at ``s_anchor = 1.0`` (the same pixel size
  ``RenderNLFPoses`` produced). Every other clone is sized by
  ``s_k = h_k / h_anchor`` to preserve the inter-person bbox-height
  ratios from the reference image, with ``target_height_clamp_min`` /
  ``target_height_clamp_max`` as a guardrail on the non-anchor scale.
  Uses additive (max) compositing so overlapping clones don't
  double-darken, plus an optional pastelization of every other clone
  to mimic the alternating per-person color scheme used by
  ``render_multi_nlf_as_images`` in ``NLFPoseExtract/nlf_render.py``
  (which lerps each saturated person-0 limb color toward white to
  obtain the lighter person-1 palette while preserving the
  warm-on-right / cool-on-left body-side semantic SCAIL was trained
  on). For >2 clones, the saturated/pastel pair cycles via
  ``k_idx % 2``, matching the upstream renderer's ``person_idx % 2``
  color-scheme assignment.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_BBOX_TYPE = "BBOXES"
_FOREGROUND_EPSILON = 4.0 / 255.0


def _bbox_from_candidate(
    candidate: np.ndarray,
    subset: Optional[np.ndarray],
    min_visible: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Return a normalized 0..1 bbox over visible keypoints, or None."""
    pts = np.asarray(candidate, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[-1] < 2:
        return None
    if subset is not None:
        sub = np.asarray(subset, dtype=np.float32).reshape(-1)
        if sub.shape[0] != pts.shape[0]:
            sub = sub[: pts.shape[0]]
        valid_mask = sub >= 0
    else:
        valid_mask = np.ones(pts.shape[0], dtype=bool)

    valid_mask &= np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    valid_mask &= (pts[:, 0] > 0) | (pts[:, 1] > 0)

    if int(valid_mask.sum()) < min_visible:
        return None

    visible = pts[valid_mask]
    x0 = float(np.min(visible[:, 0]))
    y0 = float(np.min(visible[:, 1]))
    x1 = float(np.max(visible[:, 0]))
    y1 = float(np.max(visible[:, 1]))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _coerce_canvas_size(
    dw_poses: Dict[str, Any],
    frame: Dict[str, Any],
    image_width: int,
    image_height: int,
    reference_image: Optional[torch.Tensor],
) -> Tuple[int, int]:
    """Resolve canvas (W, H) from optional inputs.

    Priority: explicit positive widgets > optional reference IMAGE shape >
    any ``H``/``W``/``image_size``/``canvas_*`` field on the DWPOSES dict
    or the per-frame dict (none exist today, but it's cheap to honour
    them in case the upstream extractor is extended). Raises
    ``ValueError`` if no source resolves the canvas.
    """
    if image_width and image_height:
        return int(image_width), int(image_height)

    if reference_image is not None and isinstance(reference_image, torch.Tensor):
        if reference_image.ndim == 4:
            _, h, w, _ = reference_image.shape
            return int(w), int(h)
        if reference_image.ndim == 3:
            h, w, _ = reference_image.shape
            return int(w), int(h)

    for source in (frame, dw_poses):
        if not isinstance(source, dict):
            continue
        for w_key, h_key in (
            ("canvas_width", "canvas_height"),
            ("W", "H"),
            ("width", "height"),
        ):
            if w_key in source and h_key in source:
                try:
                    return int(source[w_key]), int(source[h_key])
                except (TypeError, ValueError):
                    continue
        if "image_size" in source:
            try:
                size = source["image_size"]
                if len(size) == 2:
                    a, b = int(size[0]), int(size[1])
                    return b, a
            except (TypeError, ValueError):
                pass

    raise ValueError(
        "LisaBBoxesFromDWPose could not resolve canvas size. "
        "Set image_width/image_height widget values, or wire the "
        "reference_image input to the same image used for DWPose detection."
    )


class LisaBBoxesFromDWPose:
    """Extract per-person bboxes (in pixel coords) from a DWPOSES payload."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dw_poses": ("DWPOSES",),
                "frame_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1 << 20, "step": 1},
                ),
                "min_visible_keypoints": (
                    "INT",
                    {"default": 6, "min": 1, "max": 18, "step": 1},
                ),
                "padding": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                "image_width": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 1},
                ),
                "image_height": (
                    "INT",
                    {"default": 0, "min": 0, "max": 16384, "step": 1},
                ),
            },
            "optional": {
                "reference_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = (_BBOX_TYPE, "INT", "INT")
    RETURN_NAMES = ("bboxes", "canvas_width", "canvas_height")
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/Lisa"
    DESCRIPTION = (
        "Derive left-to-right pixel bboxes for each detected person in a "
        "DWPose frame. Output BBOXES bundles the boxes with the canvas "
        "size so downstream nodes don't have to be told."
    )

    def process(
        self,
        dw_poses: Dict[str, Any],
        frame_index: int,
        min_visible_keypoints: int,
        padding: float,
        image_width: int,
        image_height: int,
        reference_image: Optional[torch.Tensor] = None,
    ):
        if not isinstance(dw_poses, dict) or "poses" not in dw_poses:
            raise ValueError("dw_poses must be a DWPOSES dict with a 'poses' list.")

        poses = dw_poses["poses"]
        if frame_index < 0 or frame_index >= len(poses):
            frame_index = max(0, min(len(poses) - 1, int(frame_index)))
        frame = poses[frame_index]

        canvas_w, canvas_h = _coerce_canvas_size(
            dw_poses, frame, image_width, image_height, reference_image
        )

        candidates = np.asarray(frame["bodies"]["candidate"], dtype=np.float32)
        subsets = np.asarray(frame["bodies"]["subset"], dtype=np.float32)
        if candidates.ndim != 3:
            raise ValueError(
                f"Expected DWPose candidate of shape (N, K, 2); got {candidates.shape}."
            )

        n_persons = candidates.shape[0]
        boxes_pixel: List[Tuple[float, List[float]]] = []
        for p_idx in range(n_persons):
            normed = _bbox_from_candidate(
                candidates[p_idx], subsets[p_idx], int(min_visible_keypoints)
            )
            if normed is None:
                continue
            x0n, y0n, x1n, y1n = normed
            x0 = x0n * canvas_w
            y0 = y0n * canvas_h
            x1 = x1n * canvas_w
            y1 = y1n * canvas_h
            bw = x1 - x0
            bh = y1 - y0
            pad = float(padding)
            x0 = max(0.0, x0 - pad * bw)
            y0 = max(0.0, y0 - pad * bh)
            x1 = min(float(canvas_w), x1 + pad * bw)
            y1 = min(float(canvas_h), y1 + pad * bh)
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            cx = 0.5 * (x0 + x1)
            boxes_pixel.append((cx, [float(x0), float(y0), float(x1), float(y1)]))

        boxes_pixel.sort(key=lambda item: item[0])
        bboxes = [box for _, box in boxes_pixel]

        logging.info(
            "LisaBBoxesFromDWPose: kept %d/%d persons on %dx%d canvas",
            len(bboxes),
            n_persons,
            canvas_w,
            canvas_h,
        )

        payload = {
            "bboxes": bboxes,
            "canvas_size": (int(canvas_w), int(canvas_h)),
        }
        return (payload, int(canvas_w), int(canvas_h))


def _frame_foreground_bbox(
    frame: torch.Tensor,
) -> Optional[Tuple[int, int, int, int, torch.Tensor]]:
    """Return (sx0, sy0, sx1, sy1, mask) in source pixel coords for one frame."""
    if frame.ndim != 3:
        return None
    fg = frame.amax(dim=-1) > _FOREGROUND_EPSILON
    if not bool(fg.any()):
        return None
    rows = fg.any(dim=1)
    cols = fg.any(dim=0)
    ys = torch.nonzero(rows, as_tuple=False).flatten()
    xs = torch.nonzero(cols, as_tuple=False).flatten()
    sy0 = int(ys.min().item())
    sy1 = int(ys.max().item()) + 1
    sx0 = int(xs.min().item())
    sx1 = int(xs.max().item()) + 1
    if sx1 <= sx0 + 1 or sy1 <= sy0 + 1:
        return None
    return sx0, sy0, sx1, sy1, fg


def _drv_bbox_from_dwpose(
    driving_dw_poses: Optional[Dict[str, Any]],
    canvas_w: int,
    canvas_h: int,
    min_visible: int,
) -> Optional[Tuple[int, int, int, int]]:
    if driving_dw_poses is None or not isinstance(driving_dw_poses, dict):
        return None
    poses = driving_dw_poses.get("poses")
    if not poses:
        return None
    for frame in poses:
        try:
            cand = np.asarray(frame["bodies"]["candidate"], dtype=np.float32)
            sub = np.asarray(frame["bodies"]["subset"], dtype=np.float32)
        except (KeyError, TypeError):
            continue
        if cand.ndim != 3 or cand.shape[0] == 0:
            continue
        normed = _bbox_from_candidate(cand[0], sub[0], min_visible)
        if normed is None:
            continue
        x0n, y0n, x1n, y1n = normed
        sx0 = int(round(x0n * canvas_w))
        sy0 = int(round(y0n * canvas_h))
        sx1 = int(round(x1n * canvas_w))
        sy1 = int(round(y1n * canvas_h))
        if sx1 - sx0 < 2 or sy1 - sy0 < 2:
            continue
        return sx0, sy0, sx1, sy1
    return None


def _foot_line_y(mask: torch.Tensor) -> int:
    """Return the lowest row containing foreground (in mask-local coords)."""
    rows = mask.any(dim=1)
    ys = torch.nonzero(rows, as_tuple=False).flatten()
    if ys.numel() == 0:
        return mask.shape[0] - 1
    return int(ys.max().item())


def _resize_bilinear(
    src: torch.Tensor, mask: torch.Tensor, target_w: int, target_h: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    img = src.permute(2, 0, 1).unsqueeze(0)
    img_resized = F.interpolate(
        img, size=(target_h, target_w), mode="bilinear", align_corners=False
    )[0].permute(1, 2, 0)
    msk = mask.float().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(
        msk, size=(target_h, target_w), mode="bilinear", align_corners=False
    )[0, 0]
    mask_resized_bool = mask_resized > 0.5
    return img_resized, mask_resized_bool


_PASTELIZE_ALPHA = 0.55


def _pastelize(rgb: torch.Tensor, alpha: float = _PASTELIZE_ALPHA) -> torch.Tensor:
    """Lerp saturated RGB toward white to mimic SCAIL's person-1 palette.

    SCAIL's ``render_multi_nlf_as_images`` ships two color dictionaries:
    ``first_person_base_colors_255_dict`` (saturated, e.g. Red ``[255,20,20]``,
    Cyan ``[0,230,255]``, Pure Blue ``[0,0,255]``) and
    ``second_person_base_colors_255_dict`` which is the same warm-on-right /
    cool-on-left body-side mapping but lightened toward white (Red
    ``[255,150,150]``, Cyan ``[180,230,240]``, Pure Blue ``[120,140,255]``).
    A simple ``c' = c*(1-alpha) + alpha`` lerp with ``alpha≈0.55`` reproduces
    those second-person values almost exactly while leaving the body-side
    color semantics intact, which is what SCAIL was trained to read.

    Inputs are float tensors in ``[0, 1]``. Background pixels (≈0) are
    lifted by this lerp too, but the caller multiplies by the per-clone
    foreground mask immediately afterwards so the background returns to
    zero in the composite.
    """
    return rgb * (1.0 - alpha) + alpha


def _recolor_for_clone(rgb: torch.Tensor, k_idx: int) -> torch.Tensor:
    """Pick the SCAIL-matched color scheme for clone index ``k_idx``.

    SCAIL only trained on two schemes (saturated person-0, pastel
    person-1), so we cycle ``k_idx % 2`` exactly like
    ``render_multi_nlf_as_images``. With >2 clones at similar depth,
    persons 0 and 2 will share the saturated palette; that's the same
    behavior the upstream renderer would have produced and the only
    in-distribution choice we have without re-rendering through 3D NLF.
    """
    if k_idx % 2 == 1:
        return _pastelize(rgb)
    return rgb


# Tolerances for the same-z spread-apart pass. Tuned empirically: 5% of image
# height accommodates the small foot-line jitter in DWPose without bridging
# real depth differences (a ref person standing on a step is typically >5%
# higher); the close-fraction threshold treats a horizontal gap smaller than
# half the average bbox width as "shoulder-to-shoulder".
_SAME_Z_FEET_TOL_FRAC = 0.05
_SAME_Z_CLOSE_FRAC = 0.5


def _compute_cx_overrides(
    scaled_bboxes: List[List[float]],
    image_w: int,
    image_h: int,
    same_z_gap: float,
) -> Dict[int, float]:
    """Per-clone center-x overrides for the same-z spread-apart pass.

    Two clones are considered "same z + close" when:
      * their feet line (ty1, the bottom of the bbox) is within
        ``_SAME_Z_FEET_TOL_FRAC * image_height`` of each other;
      * the horizontal gap between them is smaller than
        ``_SAME_Z_CLOSE_FRAC * avg_bbox_width``.

    Members of a connected cluster (transitively close at same z) get their
    target center-x distributed evenly between ``half_w`` and
    ``image_w - half_w`` so at ``same_z_gap=1`` the leftmost clone's left
    edge sits at x=0 and the rightmost clone's right edge sits at x=W.
    Values in ``(0, 1)`` lerp from each clone's original center toward its
    edge target. Singletons (clones with no same-z neighbour) are left
    untouched. Returns ``{}`` when ``same_z_gap <= 0`` or fewer than two
    bboxes are present.

    ``scaled_bboxes`` is assumed sorted left-to-right by center-x.
    """
    if same_z_gap <= 0.0 or len(scaled_bboxes) < 2:
        return {}

    n = len(scaled_bboxes)
    feet_tol = float(_SAME_Z_FEET_TOL_FRAC) * float(image_h)

    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n - 1):
        a = scaled_bboxes[i]
        b = scaled_bboxes[i + 1]
        feet_a = a[3]
        feet_b = b[3]
        if abs(feet_a - feet_b) > feet_tol:
            continue
        gap_x = max(0.0, b[0] - a[2])
        avg_w = 0.5 * ((a[2] - a[0]) + (b[2] - b[0]))
        if avg_w <= 0.0:
            continue
        if gap_x > _SAME_Z_CLOSE_FRAC * avg_w:
            continue
        adj[i].append(i + 1)
        adj[i + 1].append(i)

    visited = [False] * n
    clusters: List[List[int]] = []
    for start in range(n):
        if visited[start] or not adj[start]:
            continue
        stack = [start]
        cluster: List[int] = []
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            cluster.append(v)
            stack.extend(adj[v])
        if len(cluster) >= 2:
            clusters.append(sorted(cluster))

    if not clusters:
        return {}

    overrides: Dict[int, float] = {}
    gap = float(max(0.0, min(1.0, same_z_gap)))
    for cluster in clusters:
        m = len(cluster)
        for rank, idx in enumerate(cluster):
            box = scaled_bboxes[idx]
            half_w = 0.5 * (box[2] - box[0])
            half_w = min(half_w, 0.5 * float(image_w))
            if m == 1:
                target_cx = 0.5 * float(image_w)
            else:
                t = float(rank) / float(m - 1)
                left_target = half_w
                right_target = float(image_w) - half_w
                target_cx = left_target * (1.0 - t) + right_target * t
            orig_cx = 0.5 * (box[0] + box[2])
            new_cx = orig_cx * (1.0 - gap) + target_cx * gap
            overrides[idx] = new_cx

    return overrides


def _bbox_centeredness_score(
    box: List[float],
    image_w: int,
    image_h: int,
) -> float:
    """Score a bbox by ``area_frac * x_centeredness * y_centeredness``.

    Computed in pose-canvas coords as

        score = area_frac * (1 - |cx - W/2| / (W/2))
                          * (1 - |cy - H/2| / (H/2))

    where ``area_frac = w * h / (W * H)``. A bbox sitting dead-center and
    filling the frame scores ~1.0; one in a corner or vanishingly small
    scores ~0. The multiplicative form means a bbox has to be both big
    AND centered on both axes to win.

    Shared by ``_select_top_n_bboxes`` (top-N culling) and
    ``_pick_anchor_index`` (anchor selection) so the anchor is always one
    of the boxes we would have kept after culling.
    """
    half_w = 0.5 * float(image_w)
    half_h = 0.5 * float(image_h)
    image_area = float(image_w) * float(image_h)
    if image_area <= 0.0 or half_w <= 0.0 or half_h <= 0.0:
        return 0.0
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    bw = max(0.0, x1 - x0)
    bh = max(0.0, y1 - y0)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    area_frac = (bw * bh) / image_area
    cen_x = max(0.0, 1.0 - abs(cx - half_w) / half_w)
    cen_y = max(0.0, 1.0 - abs(cy - half_h) / half_h)
    return float(area_frac * cen_x * cen_y)


def _select_top_n_bboxes(
    scaled_bboxes: List[List[float]],
    max_targets: int,
    image_w: int,
    image_h: int,
) -> List[List[float]]:
    """Pick the top-N bboxes ranked by ``_bbox_centeredness_score``.

    "Top-N" = biggest people closest to the image center on both axes;
    small off-center detections and big edge crops both get penalized.
    Stable on ties: bboxes with identical scores keep their relative input
    order. Returns a new list of at most ``max_targets`` boxes.
    """
    n = len(scaled_bboxes)
    if max_targets >= n or max_targets <= 0:
        return list(scaled_bboxes)

    image_area = float(image_w) * float(image_h)
    half_w = 0.5 * float(image_w)
    half_h = 0.5 * float(image_h)
    if image_area <= 0.0 or half_w <= 0.0 or half_h <= 0.0:
        return list(scaled_bboxes[: int(max_targets)])

    scored: List[Tuple[float, int, List[float]]] = []
    for i, box in enumerate(scaled_bboxes):
        score = _bbox_centeredness_score(box, image_w, image_h)
        scored.append((-score, i, box))

    scored.sort()
    return [box for _, _, box in scored[: int(max_targets)]]


def _pick_anchor_index(
    scaled_bboxes: List[List[float]],
    image_w: int,
    image_h: int,
) -> int:
    """Return the index of the most-centered bbox in ``scaled_bboxes``.

    "Most-centered" reuses ``_bbox_centeredness_score`` so the anchor is
    always one of the boxes ``_select_top_n_bboxes`` would have kept.
    Ties are broken on lower index. Returns ``0`` for empty or
    single-element inputs.
    """
    if len(scaled_bboxes) <= 1:
        return 0
    best_idx = 0
    best_score = _bbox_centeredness_score(scaled_bboxes[0], image_w, image_h)
    for i in range(1, len(scaled_bboxes)):
        score = _bbox_centeredness_score(scaled_bboxes[i], image_w, image_h)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


class LisaPoseDuplicateByBBoxes:
    """Composite a single dancer's rendered pose canvas onto N target bboxes.

    Anchor-first scaling: the most-centered reference bbox (max
    ``area * x_centeredness * y_centeredness`` via
    ``_pick_anchor_index``) is the anchor; its clone is drawn at
    ``s_anchor = 1.0``, i.e. the same pixel size that
    ``Render NLF Poses`` produced for the driving dancer. Every other
    clone is scaled by ``s_k = h_k / h_anchor`` to preserve the
    bbox-height ratios from the reference image, then clamped by
    ``target_height_clamp_min`` / ``target_height_clamp_max`` to bound
    pathological ratios (the clamp does not apply to the anchor itself).

    The rest of the pipeline is unchanged: scan-based per-frame
    foreground extraction, bilinear rescale, optional foot-pin
    alignment, pastelization (lerp toward white) on every other clone
    to mimic ``render_multi_nlf_as_images``'s alternating
    saturated/pastel palette without a re-render, max-compositing on a
    black canvas. The pastel transform preserves SCAIL's
    warm-on-right / cool-on-left body-side color semantic; for >2 clones
    the saturated/pastel pair cycles via ``k_idx % 2``, matching the
    upstream renderer.

    When more reference bboxes are detected than ``max_targets``, the
    top-N are picked by the same centeredness score before anchor
    selection (see ``_select_top_n_bboxes``), so the biggest most
    centered people in the reference image win rather than just the
    leftmost N.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_canvas": ("IMAGE",),
                "ref_bboxes": (_BBOX_TYPE,),
                "target_height_clamp_min": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": (
                            "Lower bound on the relative scale "
                            "(h_k / h_anchor) of non-anchor clones. The "
                            "anchor clone (the most-centered ref bbox by "
                            "area * x_centeredness * y_centeredness) is "
                            "always drawn at s=1.0, i.e. the same pixel "
                            "size Render NLF Poses produced for the "
                            "driving dancer; this widget only caps how "
                            "small the other clones can shrink relative "
                            "to the anchor. Widen toward 0.1 when the "
                            "reference image has strong depth spread "
                            "(e.g. one person far in the background)."
                        ),
                    },
                ),
                "target_height_clamp_max": (
                    "FLOAT",
                    {
                        "default": 1.3,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": (
                            "Upper bound on the relative scale "
                            "(h_k / h_anchor) of non-anchor clones. See "
                            "target_height_clamp_min; widen toward 5.0 "
                            "when a non-anchor person is much closer to "
                            "camera than the anchor."
                        ),
                    },
                ),
                "recolor_alternating": ("BOOLEAN", {"default": True}),
                "floor_pin": ("BOOLEAN", {"default": True}),
                "validate_feet": ("BOOLEAN", {"default": True}),
                "max_targets": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": (
                            "Maximum number of clones to composite. When "
                            "more reference bboxes are detected than this, "
                            "the top-N are picked by (bbox area) * "
                            "(x-centeredness) * (y-centeredness), so the "
                            "biggest people closest to the image center on "
                            "both axes win."
                        ),
                    },
                ),
                "passthrough_when_single": ("BOOLEAN", {"default": True}),
                "same_z_gap": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Spread clones apart when two or more are at the "
                            "same vertical floor (feet within ~5% of image "
                            "height) and shoulder-to-shoulder. 0.0 = leave "
                            "centers untouched, 1.0 = push outermost clones' "
                            "outer edges to the image edges, intermediate "
                            "values lerp in between."
                        ),
                    },
                ),
            },
            "optional": {
                "driving_dw_poses": ("DWPOSES",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "WanVideoWrapper/Lisa"
    DESCRIPTION = (
        "Duplicate a single dancer's pose canvas onto every detected "
        "reference-image bbox at similar depth. Output is a per-frame "
        "skeleton sequence on a black canvas with the same shape as the "
        "input pose canvas."
    )

    def process(
        self,
        pose_canvas: torch.Tensor,
        ref_bboxes: Dict[str, Any],
        target_height_clamp_min: float,
        target_height_clamp_max: float,
        recolor_alternating: bool,
        floor_pin: bool,
        validate_feet: bool,
        max_targets: int,
        passthrough_when_single: bool = True,
        same_z_gap: float = 0.0,
        driving_dw_poses: Optional[Dict[str, Any]] = None,
    ):
        if pose_canvas.ndim != 4:
            raise ValueError(
                f"pose_canvas must be (B, H, W, 3); got {tuple(pose_canvas.shape)}."
            )
        canvas = pose_canvas
        if not torch.is_floating_point(canvas):
            canvas = canvas.float() / 255.0
        canvas = canvas.detach().to("cpu")
        B, H, W, C = canvas.shape
        if C < 3:
            raise ValueError("pose_canvas must have at least 3 channels.")
        canvas = canvas[..., :3].contiguous()

        if not isinstance(ref_bboxes, dict) or "bboxes" not in ref_bboxes:
            raise ValueError(
                "ref_bboxes must be a BBOXES dict with 'bboxes' and 'canvas_size'."
            )
        bboxes_in = list(ref_bboxes.get("bboxes") or [])
        ref_canvas = ref_bboxes.get("canvas_size") or (W, H)
        ref_w, ref_h = int(ref_canvas[0]), int(ref_canvas[1])
        if ref_w <= 0 or ref_h <= 0:
            ref_w, ref_h = W, H

        sx_scale = float(W) / float(ref_w)
        sy_scale = float(H) / float(ref_h)

        scaled_bboxes: List[List[float]] = []
        for box in bboxes_in:
            x0, y0, x1, y1 = (float(v) for v in box[:4])
            x0 *= sx_scale
            x1 *= sx_scale
            y0 *= sy_scale
            y1 *= sy_scale
            x0 = max(0.0, min(float(W) - 1.0, x0))
            x1 = max(0.0, min(float(W), x1))
            y0 = max(0.0, min(float(H) - 1.0, y0))
            y1 = max(0.0, min(float(H), y1))
            if x1 - x0 < 2.0 or y1 - y0 < 2.0:
                continue
            scaled_bboxes.append([x0, y0, x1, y1])

        if not scaled_bboxes:
            logging.warning(
                "LisaPoseDuplicateByBBoxes: no usable target bboxes; passing "
                "input pose canvas through unchanged."
            )
            return (pose_canvas,)

        n_detected = len(scaled_bboxes)
        scaled_bboxes = _select_top_n_bboxes(
            scaled_bboxes, int(max_targets), int(W), int(H)
        )
        if n_detected > len(scaled_bboxes):
            logging.info(
                "LisaPoseDuplicateByBBoxes: picked top %d/%d bboxes by "
                "area * x_centeredness * y_centeredness.",
                len(scaled_bboxes),
                n_detected,
            )

        if passthrough_when_single and len(scaled_bboxes) == 1:
            logging.info(
                "LisaPoseDuplicateByBBoxes: single ref-image person detected; "
                "passing input pose canvas through unchanged "
                "(passthrough_when_single=True)."
            )
            return (pose_canvas,)

        scaled_bboxes.sort(key=lambda b: 0.5 * (b[0] + b[2]))

        clamp_min = float(min(target_height_clamp_min, target_height_clamp_max))
        clamp_max = float(max(target_height_clamp_min, target_height_clamp_max))

        anchor_idx = _pick_anchor_index(scaled_bboxes, int(W), int(H))
        anchor_box = scaled_bboxes[anchor_idx]
        h_anchor = float(anchor_box[3] - anchor_box[1])
        if h_anchor <= 0.0:
            h_anchor = 1.0

        logging.info(
            "LisaPoseDuplicateByBBoxes: anchor clone is index %d/%d "
            "(bbox=%s, h_anchor=%.1f). Anchor will be drawn at s=1.0 "
            "(Render NLF Poses native size); other clones scale by "
            "h_k/h_anchor clamped to [%.2f, %.2f].",
            anchor_idx,
            len(scaled_bboxes),
            [round(v, 1) for v in anchor_box],
            h_anchor,
            clamp_min,
            clamp_max,
        )

        cx_overrides = _compute_cx_overrides(
            scaled_bboxes, int(W), int(H), float(same_z_gap)
        )
        if cx_overrides:
            logging.info(
                "LisaPoseDuplicateByBBoxes: same_z_gap=%.2f shifts %d/%d clones "
                "(ref-image floor cluster detected).",
                float(same_z_gap),
                len(cx_overrides),
                len(scaled_bboxes),
            )

        drv_static = _drv_bbox_from_dwpose(driving_dw_poses, W, H, min_visible=6)

        out_full = self._composite_all_frames(
            canvas,
            scaled_bboxes,
            drv_static,
            clamp_min,
            clamp_max,
            anchor_idx,
            h_anchor,
            bool(recolor_alternating),
            bool(floor_pin),
            bool(validate_feet),
            cx_overrides,
        )

        out_full = out_full.clamp_(0.0, 1.0)
        return (out_full.contiguous(),)

    def _composite_all_frames(
        self,
        canvas: torch.Tensor,
        scaled_bboxes: List[List[float]],
        drv_static: Optional[Tuple[int, int, int, int]],
        clamp_min: float,
        clamp_max: float,
        anchor_idx: int,
        h_anchor: float,
        recolor_alternating: bool,
        floor_pin: bool,
        validate_feet: bool,
        cx_overrides: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        B, H, W, _ = canvas.shape
        out = torch.zeros_like(canvas)

        for f_idx in range(B):
            frame = canvas[f_idx]
            fg_info = _frame_foreground_bbox(frame)
            if fg_info is None:
                continue
            sx0, sy0, sx1, sy1, mask_full = fg_info
            if drv_static is not None:
                dsx0, dsy0, dsx1, dsy1 = drv_static
                src_x0 = max(0, min(W - 1, dsx0))
                src_y0 = max(0, min(H - 1, dsy0))
                src_x1 = max(src_x0 + 1, min(W, dsx1))
                src_y1 = max(src_y0 + 1, min(H, dsy1))
            else:
                src_x0, src_y0, src_x1, src_y1 = sx0, sy0, sx1, sy1

            crop_img = frame[src_y0:src_y1, src_x0:src_x1, :3]
            crop_mask = mask_full[src_y0:src_y1, src_x0:src_x1]
            crop_h = src_y1 - src_y0
            crop_w = src_x1 - src_x0
            if crop_h < 2 or crop_w < 2 or not bool(crop_mask.any()):
                continue

            for k_idx, box in enumerate(scaled_bboxes):
                tx0, ty0, tx1, ty1 = box
                t_h = ty1 - ty0
                if t_h <= 0:
                    continue
                if k_idx == anchor_idx:
                    s_k = 1.0
                else:
                    s_k = float(t_h) / float(h_anchor)
                    s_k = max(clamp_min, min(clamp_max, s_k))

                new_h = max(2, int(round(crop_h * s_k)))
                new_w = max(2, int(round(crop_w * s_k)))
                scaled_img, scaled_mask = _resize_bilinear(
                    crop_img, crop_mask, new_w, new_h
                )

                feet_in_target = (ty1 / float(H)) >= 0.85
                use_floor = floor_pin and (feet_in_target or not validate_feet)

                if cx_overrides is not None and k_idx in cx_overrides:
                    cx = float(cx_overrides[k_idx])
                else:
                    cx = 0.5 * (tx0 + tx1)
                if use_floor:
                    foot_y_local = _foot_line_y(scaled_mask)
                    dst_y1 = int(round(ty1))
                    dst_y0 = dst_y1 - foot_y_local - 1
                else:
                    cy = 0.5 * (ty0 + ty1)
                    dst_y0 = int(round(cy - new_h / 2.0))
                dst_x0 = int(round(cx - new_w / 2.0))

                src_y0_clip = max(0, -dst_y0)
                src_x0_clip = max(0, -dst_x0)
                src_y1_clip = new_h - max(0, (dst_y0 + new_h) - H)
                src_x1_clip = new_w - max(0, (dst_x0 + new_w) - W)
                if src_x1_clip <= src_x0_clip or src_y1_clip <= src_y0_clip:
                    continue

                place_img = scaled_img[
                    src_y0_clip:src_y1_clip, src_x0_clip:src_x1_clip, :
                ]
                place_mask = scaled_mask[
                    src_y0_clip:src_y1_clip, src_x0_clip:src_x1_clip
                ]
                place_h, place_w = place_img.shape[0], place_img.shape[1]
                if place_h == 0 or place_w == 0:
                    continue

                dy0 = max(0, dst_y0)
                dx0 = max(0, dst_x0)
                dy1 = dy0 + place_h
                dx1 = dx0 + place_w

                if recolor_alternating:
                    place_img = _recolor_for_clone(place_img, k_idx)

                place_img = place_img * place_mask.unsqueeze(-1).float()

                target_slice = out[f_idx, dy0:dy1, dx0:dx1, :]
                if target_slice.shape != place_img.shape:
                    continue
                out[f_idx, dy0:dy1, dx0:dx1, :] = torch.maximum(
                    target_slice, place_img
                )

        return out


NODE_CLASS_MAPPINGS = {
    "LisaBBoxesFromDWPose": LisaBBoxesFromDWPose,
    "LisaPoseDuplicateByBBoxes": LisaPoseDuplicateByBBoxes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LisaBBoxesFromDWPose": "Lisa BBoxes From DWPose",
    "LisaPoseDuplicateByBBoxes": "Lisa Pose Duplicate By BBoxes",
}
