# import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# import argparse, json
# import numpy as np
# import torch
# from pathlib import Path

# from cubifyanything.cubify_transformer import make_cubify_transformer
# from cubifyanything.preprocessor import Augmentor, Preprocessor
# from customize_dataset import CustomizeVideoDataset
# from time import time


# def tensor_to_np(x):
#     if torch.is_tensor(x):
#         return x.detach().cpu().numpy()
#     return np.asarray(x)


# def compute_corners_from_center_scale_R(center, dims, R):
#     # center: (3,), dims: (3,) = (dx,dy,dz), R: (3,3)
#     dx, dy, dz = dims / 2.0
#     # 8 corners in local box frame
#     corners_local = np.array([
#         [-dx, -dy, -dz], [-dx, -dy,  dz], [-dx,  dy, -dz], [-dx,  dy,  dz],
#         [ dx, -dy, -dz], [ dx, -dy,  dz], [ dx,  dy, -dz], [ dx,  dy,  dz]
#     ], dtype=np.float32)
#     return (corners_local @ R.T) + center  # (8,3)


# def project_points(corners_3d, K, RT):
#     # Robust: ensure (N,3) and guard z-divide
#     c = np.asarray(corners_3d, dtype=np.float32).reshape(-1, 3)
#     ones = np.ones((c.shape[0], 1), dtype=np.float32)
#     pts_h = np.hstack([c, ones])                                  # (N,4)
#     cam_pts = (RT.astype(np.float32) @ pts_h.T).T[:, :3]          # (N,3)
#     uvw = (K.astype(np.float32) @ cam_pts.T).T                    # (N,3)
#     uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)            # (N,2)
#     return uv


# def boxes3d_to_jsonable(instances, K, RT, boxes_key="pred_boxes_3d"):
#     if len(instances) == 0:
#         return []

#     boxes = instances.get(boxes_key)  # GeneralInstance3DBoxes
#     centers = tensor_to_np(boxes.gravity_center)   # (N,3)
#     dims    = tensor_to_np(boxes.dims)             # (N,3)
#     Rmats   = tensor_to_np(boxes.R)                # (N,3,3)

#     # optional fields
#     ids    = instances.get("gt_ids") if instances.has("gt_ids") else None
#     cats   = instances.get("categories") if instances.has("categories") else None
#     scores = tensor_to_np(instances.get("scores")) if instances.has("scores") else None

#     # corners (prefer method if exists)
#     corners_all = None
#     if hasattr(boxes, "corners"):
#         try:
#             corners_all = tensor_to_np(boxes.corners())  # (N,8,3)
#         except Exception:
#             corners_all = None

#     out = []
#     N = len(instances)
#     for i in range(N):
#         center_i, dims_i, R_i = centers[i], dims[i], Rmats[i]
#         if corners_all is not None:
#             corners_i = np.asarray(corners_all[i], dtype=np.float32).reshape(-1, 3)
#         else:
#             corners_i = compute_corners_from_center_scale_R(center_i, dims_i, R_i)

#         uv = project_points(corners_i, K, RT)
#         xmin, ymin = float(uv[:, 0].min()), float(uv[:, 1].min())
#         xmax, ymax = float(uv[:, 0].max()), float(uv[:, 1].max())

#         item = {
#             "id": str(ids[i]) if ids is not None else None,
#             "category": str(cats[i]) if cats is not None else "object",
#             "position": center_i.tolist(),
#             "scale": dims_i.tolist(),
#             "R": R_i.tolist(),
#             "corners": corners_i.tolist(),
#             "box_2d_proj": [xmin, ymin, xmax, ymax],
#             "box_2d_rend": [xmin, ymin, xmax, ymax],
#         }
#         if scores is not None:
#             item["score"] = float(scores[i])
#         out.append(item)
#     return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("dataset_path", help="Root folder containing <video_id>/<timestamp>.wide & .gt")
#     ap.add_argument("--model-path", required=True)
#     ap.add_argument("--device", default="cpu")
#     ap.add_argument("--score-thresh", type=float, default=0.25)
#     ap.add_argument("--every-nth-frame", type=int, default=None)
#     args = ap.parse_args()

#     root = Path(args.dataset_path)
#     dataset = CustomizeVideoDataset(data_path=args.dataset_path)

#     ckpt = torch.load(args.model_path, map_location=args.device)["model"]
#     backbone_dim = ckpt["backbone.0.patch_embed.proj.weight"].shape[0]
#     is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in ckpt.keys())

#     model = make_cubify_transformer(dimension=backbone_dim, depth_model=is_depth_model).eval()
#     model.load_state_dict(ckpt)
#     model = model.to(args.device)

#     augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
#     preproc = Preprocessor()
#     anchor = model.pixel_mean  # on correct device

#     if args.every_nth_frame:
#         import itertools
#         dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

#     for sample in dataset:
#         vid = str(sample["meta"]["video_id"])
#         ts  = str(int(sample["meta"]["timestamp"]))  # integer timestamp per your structure

#         wide_dir = root / f"{ts}.wide"
#         gt_dir   = root / f"{ts}.gt"
#         wide_dir.mkdir(parents=True, exist_ok=True)

#         # Use K/RT from sensor_info (already batched tensors)
#         K  = sample["sensor_info"].wide.image.K[-1].detach().cpu().numpy().astype(np.float32)  # (3,3)
#         RT = sample["sensor_info"].wide.RT[-1].detach().cpu().numpy().astype(np.float32)        # (4,4)

#         # package → preprocess → infer
#         pk = augmentor.package(sample)
#         pk = {k: {kk: v.to(anchor.device) for kk, v in sub.items()} for k, sub in pk.items()}
#         inp = preproc.preprocess([pk])

#         with torch.no_grad():
#             start_time = time()
#             pred = model(inp)[0]
#             elapsed_time = time() - start_time
#             print(f"[{ts}] Inference took {elapsed_time:.2f} seconds")
#         if pred.has("scores"):
#             pred = pred[pred.scores >= args.score_thresh]

#         items = boxes3d_to_jsonable(pred, K, RT, boxes_key="pred_boxes_3d")

#         # Write predictions as instances.json (per your spec)
#         out_path = wide_dir / "instances.json"
#         with open(out_path, "w") as f:
#             json.dump(items, f, indent=2)
#         print(f"[{vid}/{ts}] wrote {len(items)} boxes → {out_path}")

#     print("Done.")


# if __name__ == "__main__":
#     main()
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse, json
import numpy as np
import torch
from pathlib import Path
from time import time
from PIL import Image, ImageDraw, ImageFont
import clip

from cubifyanything.cubify_transformer import make_cubify_transformer
from cubifyanything.preprocessor import Augmentor, Preprocessor
from customize_dataset import CustomizeVideoDataset


# ----------------------------- utils -----------------------------

def tensor_to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_uint8_hwc(x):
    """
    Convert image tensor/ndarray to uint8 HxWx3 safely.
    Handles (C,H,W), (H,W), (H,W,C), float/integer dtypes, odd channel counts.
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)

    # layout handling
    if x.ndim == 3 and x.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
        x = np.moveaxis(x, 0, -1)
    elif x.ndim == 2:  # grayscale
        x = x[..., None]

    # dtype handling
    if np.issubdtype(x.dtype, np.floating):
        m = np.nanmax(x) if np.size(x) > 0 else 1.0
        if not np.isfinite(m) or m == 0:
            m = 1.0
        # assume 0..1 floats unless max > 1.5
        if m <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    else:
        # integers: clip to [0,255] and cast
        x = np.clip(x, 0, 255).astype(np.uint8)

    # channel handling
    if x.ndim == 3:
        c = x.shape[-1]
        if c == 1:
            x = np.repeat(x, 3, axis=-1)
        elif c > 3:
            x = x[..., :3]
    else:
        # if somehow still not 3D, force RGB
        x = np.repeat(x[..., None], 3, axis=-1)

    return x


def compute_corners_from_center_scale_R(center, dims, R):
    """
    center: (3,), dims: (dx,dy,dz) full lengths, R: (3,3)
    Returns 8 corners in the same frame as center/R, in this fixed order:
        0 (-x,-y,-z), 1 (-x,-y,+z), 2 (-x,+y,-z), 3 (-x,+y,+z),
        4 (+x,-y,-z), 5 (+x,-y,+z), 6 (+x,+y,-z), 7 (+x,+y,+z)
    """
    dx, dy, dz = dims / 2.0
    corners_local = np.array([
        [-dx, -dy, -dz], [-dx, -dy,  dz], [-dx,  dy, -dz], [-dx,  dy,  dz],
        [ dx, -dy, -dz], [ dx, -dy,  dz], [ dx,  dy, -dz], [ dx,  dy,  dz]
    ], dtype=np.float32)
    # row-vector convention
    return (corners_local @ R.T) + center  # (8,3)


def project_points(corners_3d, K, RT=None, frame="camera", RT_is_cam_to_world=True):
    """
    Project 3D points to 2D.

    corners_3d: (N,3)
    K: (3,3)
    RT: (4,4) extrinsic.
      If frame=='world':
        - If RT_is_cam_to_world=True, RT maps camera->world; we invert via X_cam = R^T (X_w - t).
        - If RT_is_cam_to_world=False, RT maps world->camera; use X_cam = R X_w + t.
      If frame=='camera': RT is ignored.
    Returns:
      uv: (N,2) with NaNs for invalid/behind-camera
      valid: (N,) boolean mask where z>0
    """
    c = np.asarray(corners_3d, dtype=np.float32).reshape(-1, 3)
    if frame == "world":
        assert RT is not None, "RT required when projecting world-frame points."
        R = RT[:3, :3].astype(np.float32)
        t = RT[:3, 3].astype(np.float32)
        if RT_is_cam_to_world:
            # world -> camera: X_cam = R^T (X_world - t)
            cam_pts = (R.T @ (c - t).T).T
        else:
            # world -> camera: X_cam = R @ X_world + t
            cam_pts = (R @ c.T + t[:, None]).T
    else:
        # already camera-frame
        cam_pts = c

    uvw = (K.astype(np.float32) @ cam_pts.T).T  # (N,3)
    z = uvw[:, 2:3]
    valid = z[:, 0] > 1e-6
    uv = np.full((uvw.shape[0], 2), np.nan, dtype=np.float32)
    good = valid
    uv[good] = uvw[good, :2] / z[good]
    return uv, valid


def edges_for_box8():
    """
    12 edges for the corner ordering defined in compute_corners_from_center_scale_R.
    """
    return [
        # x-edges (left-right)
        (0, 4), (1, 5), (2, 6), (3, 7),
        # y-edges (bottom-top)
        (0, 2), (1, 3), (4, 6), (5, 7),
        # z-edges (near-far)
        (0, 1), (2, 3), (4, 5), (6, 7),
    ]


def draw_cuboids_overlay(image_hwc_uint8, list_uvs, labels=None, confidences = None, color=None, radius=3, line_width=2, confidence_thres = 0.1):
    """
    Draw cuboids (corners + edges) on a copy of the image and, if provided,
    write the label above each box.

    image_hwc_uint8: ndarray (H,W,3) uint8
    list_uvs: list of (8,2) arrays (may contain NaNs)
    labels: optional list[str] with per-box labels (same order as list_uvs)
    """
    

    img = Image.fromarray(image_hwc_uint8)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # Pillow will fall back

    H, W = image_hwc_uint8.shape[0], image_hwc_uint8.shape[1]
    E = edges_for_box8()

    # default single color if not per-box
    default_color = (255, 0, 0)

    for idx, uv in enumerate(list_uvs):
        if confidences is not None and confidences[idx] < confidence_thres:
            continue
        col = default_color if color is None else color[idx % len(color)]

        # draw corners
        for (u, v) in uv:
            if not np.isfinite(u) or not np.isfinite(v):
                continue
            u_i = int(round(u))
            v_i = int(round(v))
            if u_i < 0 or u_i >= W or v_i < 0 or v_i >= H:
                continue
            draw.ellipse((u_i - radius, v_i - radius, u_i + radius, v_i + radius), fill=col)

        # draw edges
        for i, j in E:
            u0, v0 = uv[i]
            u1, v1 = uv[j]
            if not (np.isfinite(u0) and np.isfinite(v0) and np.isfinite(u1) and np.isfinite(v1)):
                continue
            draw.line((u0, v0, u1, v1), fill=col, width=line_width)

        # draw label (if provided), positioned above the AABB of finite points
        if labels is not None and idx < len(labels) and labels[idx]:
            finite_mask = np.isfinite(uv).all(axis=1)
            finite_uv = uv[finite_mask]
            if finite_uv.shape[0] > 0:
                xmin = float(finite_uv[:, 0].min())
                ymin = float(finite_uv[:, 1].min())
                xmax = float(finite_uv[:, 0].max())
                ymax = float(finite_uv[:, 1].max())

                text = str(labels[idx])
                # Measure text
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = draw.textsize(text, font=font)

                pad = 2
                tx = int(max(0, min(xmin, W - tw - 1)))
                ty = int(max(0, ymin - th - 4))  # try to place just above the box

                # Solid background for readability
                draw.rectangle(
                    (tx - pad, ty - pad, tx + tw + pad, ty + th + pad),
                    fill=(0, 0, 0)
                )
                # Text with a small shadow
                draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0), font=font)
                draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    return img



# --------------------- conversion + serialization ---------------------

def boxes3d_to_jsonable_and_uvs(
    instances,
    K,
    RT,
    boxes_key="pred_boxes_3d",
    frame="camera",
    RT_is_cam_to_world=True,
    img_size_wh=None,
    image=None,              # <- optional HxWx{1,3} array/tensor for cropping
    save_dir=None,           # <- optional Path/str to the <ts>.wide folder
    crop_subdir="cropped_images",
    filename_prefix="crop", model = None, preprocess = None, device="cpu", text_features=None, class_names=None, logit_scale=100.0
):
    """
    Returns:
      items: list[dict] with 3D box info + 2D AABB
      uvs_per_box: list of (8,2) projected points (NaNs for invalid)

    If `image` and `save_dir` are provided, crops each valid 2D bbox into
    `<save_dir>/<crop_subdir>/<filename_prefix>_{i:03d}_{cat}.png`.
    """
    if len(instances) == 0:
        return [], []

    boxes = instances.get(boxes_key)  # GeneralInstance3DBoxes
    centers = tensor_to_np(boxes.gravity_center)   # (N,3)
    dims    = tensor_to_np(boxes.dims)             # (N,3)
    Rmats   = tensor_to_np(boxes.R)                # (N,3,3)

    ids    = instances.get("gt_ids") if instances.has("gt_ids") else None
    cats   = instances.get("categories") if instances.has("categories") else None
    scores = tensor_to_np(instances.get("scores")) if instances.has("scores") else None

    # Try corners() if provided
    corners_all = None
    if hasattr(boxes, "corners"):
        try:
            corners_all = tensor_to_np(boxes.corners())  # (N,8,3)
        except Exception:
            corners_all = None

    items = []
    uvs_per_box = []
    N = len(instances)
    W = H = None
    if img_size_wh is not None:
        W, H = img_size_wh

    # Prepare PIL image & crop directory if requested
    pil_img = None
    crop_dir = None
    if image is not None and save_dir is not None and W is not None and H is not None:
        img_uint8 = to_uint8_hwc(image)
        # Ensure size matches provided (W,H); if not, we trust the given W,H for clamping.
        pil_img = Image.fromarray(img_uint8)
        crop_dir = Path(save_dir) / crop_subdir
        crop_dir.mkdir(parents=True, exist_ok=True)

    for i in range(N):
        label = None
        confidence = None
        center_i, dims_i, R_i = centers[i], dims[i], Rmats[i]
        if corners_all is not None:
            corners_i = np.asarray(corners_all[i], dtype=np.float32).reshape(-1, 3)
        else:
            corners_i = compute_corners_from_center_scale_R(center_i, dims_i, R_i)

        uv, valid = project_points(
            corners_i, K, RT=RT, frame=frame, RT_is_cam_to_world=RT_is_cam_to_world
        )
        uvs_per_box.append(uv)

        # AABB from valid points only
        valid_pts = uv[np.isfinite(uv).all(axis=1) & valid]
        if len(valid_pts) > 0:
            xmin, ymin = valid_pts.min(axis=0).tolist()
            xmax, ymax = valid_pts.max(axis=0).tolist()
            # optional: clamp to image bounds if provided
            if W is not None and H is not None:
                xmin = float(np.clip(xmin, 0, W - 1))
                xmax = float(np.clip(xmax, 0, W - 1))
                ymin = float(np.clip(ymin, 0, H - 1))
                ymax = float(np.clip(ymax, 0, H - 1))
        else:
            xmin = ymin = xmax = ymax = float("nan")

        # --- save crop if possible ---
        if (
            pil_img is not None and crop_dir is not None and
            np.isfinite([xmin, ymin, xmax, ymax]).all()
        ):
            # PIL crop box is (left, upper, right, lower), right/lower are exclusive
            left  = int(np.floor(xmin))
            upper = int(np.floor(ymin))
            right = int(np.ceil(xmax))
            lower = int(np.ceil(ymax))

            margin = 4
            left  = max(0, left - margin)
            upper = max(0, upper - margin)
            right = min(W, right + margin)
            lower = min(H, lower + margin)

            # Clamp again for safety
            left  = max(0, min(left,  W - 1))
            upper = max(0, min(upper, H - 1))
            right = max(0, min(right, W))
            lower = max(0, min(lower, H))

            if right > left and lower > upper:
                # Build a readable filename
                cat_str = str(cats[i]) if cats is not None else "object"
                safe_cat = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cat_str)
                fname = f"{filename_prefix}_{i:03d}_{safe_cat}.png"
                cropped_img = pil_img.crop((left, upper, right, lower))
                cropped_img.save(crop_dir / fname)

                image_input = preprocess(cropped_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    # text_features = model.encode_text(text_tokens)

                    logits = logit_scale * (image_features @ text_features.T)
                    probs = logits.softmax(dim=-1)[0].detach().cpu().numpy()
                label_idx = int(probs.argmax())
                label = class_names[label_idx]
                confidence = float(probs[label_idx])

        item = {
            "id": str(ids[i]) if ids is not None else None,
            "category": str(cats[i]) if cats is not None else "object",
            "position": center_i.tolist(),
            "scale": dims_i.tolist(),
            "R": R_i.tolist(),
            "corners": corners_i.tolist(),
            "box_2d_proj": [xmin, ymin, xmax, ymax],
            "box_2d_rend": [xmin, ymin, xmax, ymax],
            "label": label,
            "confidence": confidence,
        }
        if scores is not None:
            item["score"] = float(scores[i])
        

        items.append(item)

    return items, uvs_per_box



# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_path", help="Root folder containing <video_id>/<timestamp>.wide & .gt")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--every-nth-frame", type=int, default=None)
    ap.add_argument("--room_map_path", default="object_room_map.json",
                    help="Path to the object_room_map.json file")
    ap.add_argument("--confidence-thres", type=float, default=0.1,
                    help="Minimum confidence to consider a box valid for cropping")
    args = ap.parse_args()

    root = Path(args.dataset_path)
    dataset = CustomizeVideoDataset(data_path=args.dataset_path)


    ckpt = torch.load(args.model_path, map_location=args.device)["model"]
    backbone_dim = ckpt["backbone.0.patch_embed.proj.weight"].shape[0]
    is_depth_model = any(k.startswith("backbone.0.patch_embed_depth.") for k in ckpt.keys())
    with open(args.room_map_path, "r") as f:
        room_lookup = json.load(f)

    model = make_cubify_transformer(dimension=backbone_dim, depth_model=is_depth_model).eval()
    model.load_state_dict(ckpt)
    model = model.to(args.device)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    model_clip, preprocess = clip.load("ViT-B/32", device=device)
    class_names = list(room_lookup.keys())
    # prompted = [f"a photo of a {c}" for c in class_names]

    model_clip.eval()
    with torch.no_grad():
        text_tokens = clip.tokenize(class_names).to(device)
        text_features = model_clip.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = float(model_clip.logit_scale.exp().detach().cpu().item()) if hasattr(model_clip, "logit_scale") else 100.0

    augmentor = Augmentor(("wide/image", "wide/depth") if is_depth_model else ("wide/image",))
    preproc = Preprocessor()
    anchor = model.pixel_mean  # device anchor

    if args.every_nth_frame:
        import itertools
        dataset = itertools.islice(dataset, 0, None, args.every_nth_frame)

    for sample in dataset:
        vid = str(sample["meta"]["video_id"])
        ts  = str(int(sample["meta"]["timestamp"]))  # integer timestamp

        wide_dir = root / f"{ts}.wide"
        gt_dir   = root / f"{ts}.gt"
        wide_dir.mkdir(parents=True, exist_ok=True)

        # Input image (C,H,W) or other -> robust H,W,C uint8
        image_any = sample["wide"]["image"][-1]
        image_hwc = to_uint8_hwc(image_any)

        # Save the raw image for reference
        Image.fromarray(image_hwc).save(wide_dir / "image.png")

        # Camera intrinsics (3x3)
        K  = sample["sensor_info"].wide.image.K[-1].detach().cpu().numpy().astype(np.float32)
        # Extrinsics (4x4) — NOT used for predicted boxes in camera frame
        RT = sample["sensor_info"].wide.RT[-1].detach().cpu().numpy().astype(np.float32)

        # package → preprocess → infer
        pk = augmentor.package(sample)
        pk = {k: {kk: v.to(anchor.device) for kk, v in sub.items()} for k, sub in pk.items()}
        inp = preproc.preprocess([pk])

        with torch.no_grad():
            start_time = time()
            pred = model(inp)[0]
            

        if pred.has("scores"):
            pred = pred[pred.scores >= args.score_thresh]

        # Convert boxes + compute 2D projections (camera frame for predictions)
        W, H = sample["sensor_info"].wide.image.size  # (W,H)
        items, uvs_per_box = boxes3d_to_jsonable_and_uvs(
            pred, K, RT=None, boxes_key="pred_boxes_3d",
            frame="camera", RT_is_cam_to_world=True,
            img_size_wh=(W, H), image =image_hwc, save_dir = wide_dir, crop_subdir = 'cropped_images', filename_prefix=f"{ts}", model=model_clip, preprocess=preprocess, device=device, text_features = text_features, class_names=class_names, logit_scale = logit_scale
        )
        elapsed_time = time() - start_time
        print(f"[{ts}] Inference took {elapsed_time:.2f} seconds")

        labels = [(it.get("label") or it.get("category") or f"box_{i}") for i, it in enumerate(items)]
        confidences = [it.get("confidence", 0.0) for it in items]


        # Write predictions as instances.json
        out_path = wide_dir / "instances.json"
        with open(out_path, "w") as f:
            json.dump(items, f, indent=2)
        print(f"[{vid}/{ts}] wrote {len(items)} boxes → {out_path}")

        # ---------------- sanity-check overlay ----------------
        # Per-box colors (deterministic)
        overlay_colors = [
            tuple(int(c) for c in np.random.default_rng(i).integers(64, 255, size=3))
            for i in range(len(uvs_per_box))
        ]
        img_overlay = draw_cuboids_overlay(image_hwc, uvs_per_box, labels = labels, confidences=confidences, color=overlay_colors, radius=3, line_width=2, confidence_thres=args.confidence_thres)
        img_overlay.save(wide_dir / "overlay.png")
        print(f"[{vid}/{ts}] wrote overlay → {wide_dir/'overlay.png'}")

    print("Done.")


if __name__ == "__main__":
    main()

