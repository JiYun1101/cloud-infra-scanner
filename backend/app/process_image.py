# file: process_image.py
from __future__ import annotations

import os
import json
from typing import Tuple, List, Optional, Dict, Deque
from collections import deque

import cv2
import numpy as np


# ---------- 파일 유틸 ----------
def imread_unicode(path: str):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, img: np.ndarray) -> None:
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    buf.tofile(path)


# ---------- 기하 유틸 ----------
def _merge_similar(rects: List[Tuple[int, int, int, int]], threshold=40):
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[0], r[1]))
    merged: List[List[int]] = []
    for (x, y, w, h) in rects:
        if not merged:
            merged.append([x, y, w, h]); continue
        X, Y, W, H = merged[-1]
        if (abs(x - X) <= threshold and abs(y - Y) <= threshold and
            abs(w - W) <= threshold and abs(h - H) <= threshold):
            nx = min(x, X); ny = min(y, Y)
            nx2 = max(x + w, X + W); ny2 = max(y + h, Y + H)
            merged[-1] = [nx, ny, nx2 - nx, ny2 - ny]
        else:
            merged.append([x, y, w, h])
    return [tuple(m) for m in merged]

def _nms(rects: List[Tuple[int, int, int, int]], iou_thresh=0.3):
    if not rects:
        return []
    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=np.float32)
    scores = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order, 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        # correct per-j area computation (shape == order.shape)
        area_j = (boxes[order, 2] - boxes[order, 0]) * (boxes[order, 3] - boxes[order, 1])
        iou = inter / (area_i + area_j - inter + 1e-6)
        order = order[1:][iou[1:] < iou_thresh]
    return [rects[i] for i in keep]

def _is_contained_in(small: Tuple[int,int,int,int], big: Tuple[int,int,int,int], tol: int = 2) -> bool:
    sx, sy, sw, sh = small
    bx, by, bw, bh = big
    return (
        sx >= bx + tol and
        sy >= by + tol and
        sx + sw <= bx + bw - tol and
        sy + sh <= by + bh - tol
    )

def _intersects(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)

def _expand_rect(r: Tuple[int,int,int,int], dx: int, dy: int) -> Tuple[int,int,int,int]:
    x, y, w, h = r
    return (x - dx, y - dy, w + 2*dx, h + 2*dy)

def _inflate_rect(r: Tuple[int,int,int,int], pad: int, W: Optional[int]=None, H: Optional[int]=None) -> Tuple[int,int,int,int]:
    x, y, w, h = r
    nx, ny = x - pad, y - pad
    nw, nh = w + 2*pad, h + 2*pad
    if W is not None and H is not None:
        nx = max(0, nx); ny = max(0, ny)
        nx2 = min(W, nx + nw); ny2 = min(H, ny + nh)
        nw = max(1, nx2 - nx); nh = max(1, ny2 - ny)
    return (int(nx), int(ny), int(nw), int(nh))

def _shrink_rect(r: Tuple[int, int, int, int], pad: int) -> Tuple[int, int, int, int]:
    x, y, w, h = r
    return (x + pad, y + pad, max(1, w - 2*pad), max(1, h - 2*pad))

def _intersects_frame_band(r: Tuple[int,int,int,int], f: Tuple[int,int,int,int], band: int) -> bool:
    if band <= 0:
        return False
    fx, fy, fw, fh = f
    outer = (fx, fy, fw, fh)
    inner = _shrink_rect(outer, band)
    return _intersects(r, outer) and not _is_contained_in(r, inner, tol=0)

def _contains_point(r: Tuple[int,int,int,int], x: float, y: float) -> bool:
    rx, ry, rw, rh = r
    return (rx <= x <= rx+rw) and (ry <= y <= ry+rh)


# ---------- 텍스트 라벨 필터 ----------
def _is_textish(r: Tuple[int,int,int,int], W: int, H: int) -> bool:
    _, _, w, h = r
    if w <= 0 or h <= 0:
        return True
    ar = w / float(h)
    too_wide = ar >= 1.7
    too_flat = h <= max(14, int(0.028 * H))
    long_thin = ar >= 1.4 and h <= max(18, int(0.035 * H))
    return too_wide or too_flat or long_thin


# ---------- 노란 박스 검출 ----------
def _detect_yellow_boxes(img: np.ndarray) -> List[Tuple[int,int,int,int]]:
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    y_lower = np.array([5, 35, 95], dtype=np.uint8)
    y_upper = np.array([70, 255, 255], dtype=np.uint8)
    ymask = cv2.inRange(hsv, y_lower, y_upper)
    ymask = cv2.morphologyEx(ymask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=2)

    cand: List[Tuple[int,int,int,int]] = []
    cnts, _ = cv2.findContours(ymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w*h < 150: continue
        cand.append((x,y,w,h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(gray, (5,5), 0), 50, 150)
    e = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)
    cnts2, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts2:
        peri = cv2.arcLength(c, True)
        if peri < 28: continue
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx): continue
        x, y, w, h = cv2.boundingRect(approx)
        if w*h < 450: continue
        cand.append((x,y,w,h))

    cand = _merge_similar(cand, 16)

    def _bgr2lab_vec(bgr):
        chip = np.uint8([[bgr]])
        return cv2.cvtColor(chip, cv2.COLOR_BGR2LAB)[0,0].astype(np.float32)
    refs = [
        _bgr2lab_vec((0,200,255)), _bgr2lab_vec((0,192,255)), _bgr2lab_vec((0,180,255)),
        _bgr2lab_vec((10,190,240)), _bgr2lab_vec((0,165,255)),
        _bgr2lab_vec((0,153,255)), _bgr2lab_vec((0,215,255)),
    ]

    def ring_score(rect: Tuple[int,int,int,int], off: int, t: int = 2) -> float:
        x, y, w, h = rect
        if off >= 0:
            x = x + off; y = y + off; w = max(1, w - 2*off); h = max(1, h - 2*off)
        else:
            pad = -off
            x, y, w, h = _inflate_rect((x,y,w,h), pad, W, H)
        if w <= 2*t or h <= 2*t: return 0.0
        roi_hsv = hsv[y:y+h, x:x+w]
        roi_lab = lab[y:y+h, x:x+w].astype(np.float32)
        mask_hsv = cv2.inRange(roi_hsv, y_lower, y_upper)
        dmin = None
        for ref in refs:
            d = (roi_lab - ref)**2
            d = np.sqrt(d[...,0] + d[...,1] + d[...,2])
            dmin = d if dmin is None else np.minimum(dmin, d)
        mask_lab = (dmin < 38).astype(np.uint8) * 255
        ym = cv2.bitwise_or(mask_hsv, mask_lab)
        border_mask = np.zeros((h, w), np.uint8)
        border_mask[:t, :] = 1; border_mask[-t:, :] = 1
        border_mask[:, :t] = 1; border_mask[:, -t:] = 1
        border_pixels = int(np.count_nonzero(border_mask))
        yellow_on_border = int(np.count_nonzero(cv2.bitwise_and(ym, ym, mask=border_mask)))
        return yellow_on_border / max(1, border_pixels)

    def corner_check(rect: Tuple[int,int,int,int]) -> bool:
        x, y, w, h = rect
        offs = [0,1,2,3]; strong = 0
        for o in offs:
            for (cx, cy) in [(x+o,y+o),(x+w-1-o,y+o),(x+o,y+h-1-o),(x+w-1-o,y+h-1-o)]:
                cx0, cy0 = max(0,cx-1), max(0,cy-1)
                cx1, cy1 = min(W, cx+2), min(H, cy+2)
                patch = img[cy0:cy1, cx0:cx1]
                if patch.size == 0:
                    continue
                patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                patch_mask = cv2.inRange(patch_hsv, np.array([5,35,95],np.uint8), np.array([70,255,255],np.uint8))
                if np.count_nonzero(patch_mask) >= 2:
                    strong += 1
        return strong >= 4

    rects: List[Tuple[int,int,int,int]] = []
    min_s = max(18, int(0.010 * min(W, H)))
    max_s = min(int(0.30 * max(W, H)), 420)
    for r in cand:
        x, y, w, h = r
        if w < min_s or h < min_s or w > max_s or h > max_s: continue
        ar = w / float(h)
        if not (0.60 <= ar <= 1.80): continue
        scores = [ring_score(r, off, t=2) for off in range(-4, 5)]
        ratio = float(np.max(scores)) if scores else 0.0
        if ratio < 0.08 and not corner_check(r): continue
        rects.append(r)
    return _nms(_merge_similar(rects, 14), iou_thresh=0.30)


# ---------- 화살표 & 라인 유틸 ----------
def _boxes_container_flags(boxes: List[Tuple[int,int,int,int]]) -> List[bool]:
    n = len(boxes)
    flags = [False]*n
    for i in range(n):
        xi, yi, wi, hi = boxes[i]
        ai = wi*hi
        for j in range(n):
            if i == j: continue
            xj, yj, wj, hj = boxes[j]
            aj = wj*hj
            if aj >= ai: continue
            if _is_contained_in((xj, yj, wj, hj), (xi, yi, wi, hi), tol=3):
                flags[i] = True
                break
    return flags

def _ray_aabb(origin: Tuple[float,float], d: Tuple[float,float], rect: Tuple[int,int,int,int]) -> Optional[float]:
    ox, oy = origin; dx, dy = d
    x, y, w, h = rect
    rx1, rx2 = x, x + w; ry1, ry2 = y, y + h
    tmin, tmax = -1e18, 1e18
    if abs(dx) < 1e-8:
        if ox < rx1 or ox > rx2: return None
    else:
        tx1 = (rx1 - ox) / dx; tx2 = (rx2 - ox) / dx
        tmin = max(tmin, min(tx1, tx2)); tmax = min(tmax, max(tx1, tx2))
    if abs(dy) < 1e-8:
        if oy < ry1 or oy > ry2: return None
    else:
        ty1 = (ry1 - oy) / dy; ty2 = (ry2 - oy) / dy
        tmin = max(tmin, min(ty1, ty2)); tmax = min(tmax, max(ty1, ty2))
    if tmax < max(0.0, tmin): return None
    t = tmin if tmin > 0 else tmax
    return t if t is not None and t > 0 else None

def _acute_tip_and_dir_from_poly(cnt: np.ndarray) -> Optional[Tuple[Tuple[float,float], Tuple[float,float]]]:
    if cnt is None or len(cnt) < 3:
        return None
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) < 3:
        return None
    pts = cv2.convexHull(approx).reshape(-1, 2).astype(np.float32)
    m = len(pts)
    best_i, best_ang = -1, 1e9
    for i in range(m):
        p0 = pts[(i-1) % m]; p1 = pts[i]; p2 = pts[(i+1) % m]
        v1 = p0 - p1; v2 = p2 - p1
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: continue
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
        if ang < best_ang:
            best_ang, best_i = ang, i
    if best_i < 0 or best_ang > 130:
        return None
    tip = pts[best_i]
    base_mid = (pts[(best_i-1) % m] + pts[(best_i+1) % m]) / 2.0
    d = tip - base_mid
    n = np.linalg.norm(d)
    if n < 1e-6:
        return None
    d = (d / n).astype(np.float32)
    return (tuple(map(float, tip)), (float(d[0]), float(d[1])))

def _extract_arrow_heads(img: np.ndarray) -> List[Tuple[Tuple[float,float], Tuple[float,float]]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.inRange(hsv, np.array([0, 0, 60], np.uint8), np.array([180, 80, 245], np.uint8))
    purple = cv2.inRange(hsv, np.array([120, 35, 40], np.uint8), np.array([175, 255, 255], np.uint8))  # widened
    mask_color = cv2.bitwise_or(gray, purple)

    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(G, (3,3), 0), 50, 140)  # slightly softer
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), 1)

    mask = cv2.bitwise_or(mask_color, edges)

    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    heads: List[Tuple[Tuple[float,float], Tuple[float,float]]] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2 or area > 20000:
            continue
        td = _acute_tip_and_dir_from_poly(c)
        if td is not None:
            heads.append(td)
    return heads

def _extract_heads_near_boxes(img: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> List[Tuple[Tuple[float,float], Tuple[float,float], int]]:
    H, W = img.shape[:2]
    heads: List[Tuple[Tuple[float,float], Tuple[float,float], int]] = []
    for j, (x, y, w, h) in enumerate(boxes):
        ring_px = max(6, int(0.12 * min(w, h)))
        outer = _inflate_rect((x, y, w, h), ring_px, W, H)
        inner = _inflate_rect((x, y, w, h), 2, W, H)
        ox, oy, ow, oh = outer
        roi = img[oy:oy+oh, ox:ox+ow]
        local_heads = _extract_arrow_heads(roi)
        if not local_heads:
            continue
        cx, cy = x + w/2.0, y + h/2.0
        for (tip, d) in local_heads:
            gx, gy = tip[0] + ox, tip[1] + oy
            if (inner[0] <= gx <= inner[0]+inner[2]) and (inner[1] <= gy <= inner[1]+inner[3]):
                continue
            v_in = np.array([cx - gx, cy - gy], dtype=np.float32)
            if np.dot(v_in, np.array(d, dtype=np.float32)) <= 0:
                continue
            heads.append(((float(gx), float(gy)), (float(d[0]), float(d[1])), j))
    return heads


# ---------- 라인 경로 ----------
def _make_line_mask(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (line_mask, strong_edges).
    line_mask: relaxed union of colored lines and edges to avoid missing faint segments.
    strong_edges: plain Canny edges used for path quality check.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(hsv, np.array([120, 35, 35], np.uint8), np.array([175, 255, 255], np.uint8))
    gray = cv2.inRange(hsv, np.array([0, 0, 35], np.uint8), np.array([180, 45, 210], np.uint8))
    color_union = cv2.bitwise_or(gray, purple)

    G = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    strong = cv2.Canny(cv2.GaussianBlur(G, (3, 3), 0), 70, 170)

    # relaxed: OR with strong edges, then clean up
    m = cv2.bitwise_or(color_union, strong)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    m = cv2.dilate(m, k, iterations=1)

    return (m > 0).astype(np.uint8), (strong > 0).astype(np.uint8)


def _snap_to_mask(mask: np.ndarray, x: float, y: float, r: int = 4) -> Optional[Tuple[int,int]]:
    H, W = mask.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    best = None; best_d2 = 1e18
    for yy in range(max(0,cy-r), min(H, cy+r+1)):
        for xx in range(max(0,cx-r), min(W, cx+r+1)):
            if mask[yy, xx]:
                d2 = (xx - x)**2 + (yy - y)**2
                if d2 < best_d2:
                    best_d2, best = d2, (xx, yy)
    return best


def _trace_polyline(mask: np.ndarray,
                    tip: Tuple[float,float],
                    d: Tuple[float,float],
                    from_rect: Tuple[int,int,int,int],
                    max_steps=6000) -> List[Tuple[int,int]]:
    H, W = mask.shape[:2]
    snapped = _snap_to_mask(mask, tip[0], tip[1], r=5)
    if snapped is None:
        return []
    x, y = float(snapped[0]), float(snapped[1])

    vx, vy = -d[0], -d[1]
    path: List[Tuple[int,int]] = []
    visited = set()
    for _ in range(max_steps):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= W or iy >= H: break
        if not mask[iy, ix]: break
        path.append((ix, iy))
        visited.add((ix, iy))
        fx, fy, fw, fh = from_rect
        band = 3
        inner = (fx+band, fy+band, max(1,fw-2*band), max(1,fh-2*band))
        if (fx <= ix <= fx+fw and fy <= iy <= fy+fh) and not _is_contained_in((ix,iy,1,1), inner, tol=0):
            break
        best: Optional[Tuple[int,int]] = None
        best_dp = -1e9
        for ddy in (-1,0,1):
            for ddx in (-1,0,1):
                if ddx==0 and ddy==0: continue
                nx, ny = ix+ddx, iy+ddy
                if 0 <= nx < W and 0 <= ny < H and mask[ny, nx]:
                    if (nx, ny) in visited: continue
                    dp = ddx*vx + ddy*vy
                    if dp > best_dp:
                        best_dp, best = dp, (nx, ny)
        if best is None:
            break
        bx, by = best
        vx, vy = (bx - ix), (by - iy)
        nrm = (vx*vx + vy*vy) ** 0.5 or 1.0
        vx, vy = vx/nrm, vy/nrm
        x, y = bx, by
    if len(path) >= 4:
        path = _rdp(path, eps=1.5)
    return path


def _rdp(points: List[Tuple[int,int]], eps: float) -> List[Tuple[int,int]]:
    if len(points) <= 2: return points
    def perp(a, b, p):
        ax, ay = a; bx, by = b; px, py = p
        if a == b: return np.hypot(px-ax, py-ay)
        t = ((px-ax)*(bx-ax)+(py-ay)*(by-ay))/(((bx-ax)**2+(by-ay)**2) or 1)
        t = max(0, min(1, t))
        cx, cy = ax + t*(bx-ax), ay + t*(by-ay)
        return np.hypot(px-cx, py-cy)
    dmax, idx = 0.0, 0
    for i in range(1, len(points)-1):
        d = perp(points[0], points[-1], points[i])
        if d > dmax: dmax, idx = d, i
    if dmax > eps:
        left = _rdp(points[:idx+1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _to_xy_pairs(seq) -> Optional[List[List[int]]]:
    try:
        pairs: List[List[int]] = []
        for p in seq:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pairs.append([int(p[0]), int(p[1])])
            else:
                return None
        return pairs if len(pairs) >= 2 else None
    except Exception:
        return None


def _path_ok(path: List[Tuple[int, int]], strong_edges: np.ndarray) -> bool:
    if len(path) < 6:
        return False
    H, W = strong_edges.shape[:2]
    on_strong = 0
    for (x, y) in path:
        if 0 <= x < W and 0 <= y < H and strong_edges[y, x]:
            on_strong += 1
    ratio = on_strong / float(len(path))
    if ratio < 0.30:  # slightly relaxed
        return False
    ax, ay = path[0]
    bx, by = path[-1]
    straight = max(1.0, float(np.hypot(bx - ax, by - ay)))
    if len(path) > 6 and (len(path) / straight) > 5.5:
        return False
    return True


# ---------- 보강: 스켈레톤 기반 경로(화살표가 없는 경우) ----------
def _skeletonize(bin_mask: np.ndarray, max_iter: int = 512) -> np.ndarray:
    img = (bin_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(max_iter):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _anchors_for_box(skel: np.ndarray, rect: Tuple[int,int,int,int], W: int, H: int, max_k: int = 6) -> List[Tuple[int,int]]:
    x, y, w, h = rect
    cx, cy = int(x + w/2), int(y + h/2)
    anchors: List[Tuple[int,int]] = []
    for band in (3, 5, 8, 12):  # search outward
        outer = _inflate_rect(rect, band, W, H)
        inner = _inflate_rect(rect, 2, W, H)
        ox, oy, ow, oh = outer
        roi = skel[oy:oy+oh, ox:ox+ow]
        if roi.size == 0:
            continue
        yy, xx = np.where(roi > 0)
        cand = [(int(ox+xxi), int(oy+yyi)) for xxi, yyi in zip(xx, yy)
                if not (inner[0] <= ox+xxi <= inner[0]+inner[2] and inner[1] <= oy+yyi <= inner[1]+inner[3])]
        if not cand:
            continue
        # pick evenly by angle
        angs = np.array([np.arctan2(py-cy, px-cx) for (px, py) in cand])
        bins = np.linspace(-np.pi, np.pi, max_k+1)
        selected = []
        for i in range(len(bins)-1):
            s = [cand[k] for k, a in enumerate(angs) if bins[i] <= a < bins[i+1]]
            if s:
                # farthest from center → likely true connector
                s.sort(key=lambda p: -(p[0]-cx)**2 - (p[1]-cy)**2)
                selected.append(s[0])
        anchors.extend(selected)
        if len(anchors) >= max_k:
            break
    # unique
    uniq = []
    seen = set()
    for p in anchors:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq[:max_k]


def _inside_ring(rect: Tuple[int,int,int,int], pt: Tuple[int,int], band: int = 3) -> bool:
    x, y, w, h = rect
    inner = (x+band, y+band, max(1,w-2*band), max(1,h-2*band))
    px, py = pt
    return (x <= px <= x+w and y <= py <= y+h) and not _is_contained_in((px,py,1,1), inner, tol=0)


def _bfs_on_skeleton(skel: np.ndarray,
                     strong_edges: np.ndarray,
                     start: Tuple[int,int],
                     boxes: List[Tuple[int,int,int,int]],
                     from_idx: int,
                     max_vis: int = 40000) -> Optional[Tuple[List[Tuple[int,int]], int]]:
    H, W = skel.shape[:2]
    q: Deque[Tuple[int,int]] = deque()
    q.append(start)
    visited = np.zeros_like(skel, dtype=np.uint8)
    parents: Dict[Tuple[int,int], Tuple[int,int]] = {}
    visited[start[1], start[0]] = 1

    while q and max_vis > 0:
        max_vis -= 1
        x, y = q.popleft()
        # stop: hit ring of another box
        for j, r in enumerate(boxes):
            if j == from_idx:
                continue
            if _inside_ring(r, (x, y), band=3):
                # reconstruct
                path = [(x, y)]
                cur = (x, y)
                while cur in parents:
                    cur = parents[cur]
                    path.append(cur)
                path.reverse()
                if _path_ok(path, strong_edges):
                    return path, j
                else:
                    return None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H and skel[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    parents[(nx, ny)] = (x, y)
                    q.append((nx, ny))
    return None


# ---------- 합집합(내포 금지) ----------
def _suppress_nested(rects: List[Tuple[int,int,int,int]], keep_inner: bool = True, tol: int = 3) -> List[Tuple[int,int,int,int]]:
    n = len(rects)
    keep = [True]*n
    for i in range(n):
        for j in range(n):
            if i == j: continue
            ri, rj = rects[i], rects[j]
            if _is_contained_in(ri, rj, tol):
                if keep_inner: keep[j] = False
                else: keep[i] = False
    return [rects[k] for k in range(n) if keep[k]]

def _center(r: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x,y, w,h = r
    return (x + w/2.0, y + h/2.0)

def _union_targets(primary: List[Tuple[int,int,int,int]],
                   secondary: List[Tuple[int,int,int,int]],
                   tol_center: float = 12.0,
                   tol_contain: int = 3) -> List[Tuple[int,int,int,int]]:
    out = list(primary)
    centers = [_center(r) for r in primary]
    for r in secondary:
        if any(_is_contained_in(r, y, tol_contain) for y in primary): continue
        if any(_is_contained_in(y, r, tol_contain) for y in primary): continue
        cx, cy = _center(r)
        if any(_contains_point(y, cx, cy) for y in primary): continue
        if any((cx-cx0)**2 + (cy-cy0)**2 <= tol_center*tol_center for (cx0,cy0) in centers): continue
        out.append(r); centers.append((cx, cy))
    return out


# ---------- 메인 ----------
def process_image(input_path: str, output_path: str) -> str:
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    image = imread_unicode(input_path)
    if image is None:
        raise ValueError(f"Cannot read image from {input_path}. Check file path or type.")

    H, W = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frames: List[Tuple[int,int,int,int]] = []
    resources: List[Tuple[int,int,int,int]] = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri == 0: continue
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        area = cv2.contourArea(cnt)
        if w * h < 500 or w < 15 or h < 15: continue
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > 2000 and max(w / h, h / w) <= 2.0:
                frames.append((x, y, w, h)); continue
        if 25 < w < 180 and 25 < h < 180 and area > 120:
            circularity = 4 * np.pi * area / (peri ** 2) if peri else 0.0
            if circularity > 0.55 or len(approx) >= 5:
                resources.append((x, y, w, h))

    frames = _nms(_merge_similar(frames, threshold=40), iou_thresh=0.20)
    resources = _nms(_merge_similar(resources, threshold=25), iou_thresh=0.30)
    resources = [r for r in resources if not _is_textish(r, W, H)]
    res2 = []
    for r in resources:
        near_edge = False
        for (fx, fy, fw, fh) in frames:
            band = max(4, int(0.02 * min(fw, fh)))
            if _intersects_frame_band(r, (fx, fy, fw, fh), band):
                near_edge = True; break
        if not near_edge: res2.append(r)
    resources = res2

    yellow_boxes = _detect_yellow_boxes(image)
    yellow_boxes = _suppress_nested(yellow_boxes, keep_inner=True, tol=3)

    square_like = []
    for (rx, ry, rw, rh) in resources:
        ar = rw / float(rh)
        if 0.70 <= ar <= 1.45 and min(rw, rh) >= 28:
            pad = max(3, int(0.06 * min(rw, rh)))
            square_like.append(_inflate_rect((rx, ry, rw, rh), pad, W, H))

    fallback_candidates = resources + square_like
    yellow_targets = _union_targets(yellow_boxes, fallback_candidates, tol_center=12.0, tol_contain=3)

    container_flags = _boxes_container_flags(yellow_targets)

    diag_limit = float(np.hypot(W, H)) * 0.75

    def hit_yellow(origin: Tuple[float,float], d: Tuple[float,float], prefer_inner: bool = True) -> Optional[int]:
        hits: List[Tuple[float,int]] = []
        for i, r in enumerate(yellow_targets):
            t = _ray_aabb(origin, d, r)
            if t is None or t <= 0 or t > diag_limit: continue
            hits.append((t, i))
        if not hits: return None
        hits.sort(key=lambda x: x[0])
        if prefer_inner:
            for _, i in hits:
                if not container_flags[i]: return i
        return hits[0][1]

    # 화살촉 수집 (로컬 + 전역)
    local_heads = _extract_heads_near_boxes(image, yellow_targets)
    global_heads = _extract_arrow_heads(image)
    arrow_heads_tagged: List[Tuple[Tuple[float,float], Tuple[float,float], Optional[int]]] = []
    arrow_heads_tagged.extend(local_heads)
    for (tip, d) in global_heads:
        arrow_heads_tagged.append((tip, d, None))

    # 라인 마스크
    line_mask, strong_edges = _make_line_mask(image)

    # 간선 구성(실제 경로가 있을 때만)
    edges_yellow: List[Tuple[int,int]] = []
    edges_polyline: List[dict] = []
    seen = set()
    for (tip, d, maybe_to) in arrow_heads_tagged:
        off = 3.0
        to_i = maybe_to if maybe_to is not None else hit_yellow((tip[0] + d[0]*off, tip[1] + d[1]*off), d, True)
        fr_i = hit_yellow((tip[0] - d[0]*off, tip[1] - d[1]*off), (-d[0], -d[1]), True)
        if fr_i is None or to_i is None or fr_i == to_i:
            continue

        path = _trace_polyline(line_mask, tip, d, yellow_targets[fr_i])
        if len(path) < 6 or not _path_ok(path, strong_edges):
            continue
        px, py = path[-1]
        fx, fy, fw, fh = yellow_targets[fr_i]
        band = 3
        inner = (fx+band, fy+band, max(1,fw-2*band), max(1,fh-2*band))
        if not (fx <= px <= fx+fw and fy <= py <= fy+fh) or _is_contained_in((px,py,1,1), inner, tol=0):
            continue

        key = (fr_i, to_i)
        if key in seen:
            continue
        seen.add(key)
        edges_yellow.append(key)
        pairs = _to_xy_pairs(path)
        if pairs:
            edges_polyline.append({"from": int(fr_i), "to": int(to_i), "polyline": pairs})

    # ---------------- Fallback: 스켈레톤 기반 연결 ----------------
    if len(edges_polyline) == 0:
        skel = _skeletonize(line_mask)
        for i, rect in enumerate(yellow_targets):
            anchors = _anchors_for_box(skel, rect, W, H, max_k=6)
            if not anchors:
                continue
            for a in anchors:
                res = _bfs_on_skeleton(skel, strong_edges, a, yellow_targets, from_idx=i)
                if not res:
                    continue
                path, j = res
                if i == j:
                    continue
                key = (j, i)  # path ran from other box ring into this ring; flip to (from=j, to=i)
                if key in seen:
                    continue
                seen.add(key)
                edges_yellow.append(key)
                pairs = _to_xy_pairs(path)
                if pairs:
                    edges_polyline.append({"from": int(key[0]), "to": int(key[1]), "polyline": pairs})

    # 주석 이미지
    annotated = image.copy()
    for (x, y, w, h) in yellow_targets:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 215, 255), 2)
    for e in edges_polyline:
        pts = np.array(e["polyline"], dtype=np.int32).reshape(-1,1,2)
        cv2.polylines(annotated, [pts], False, (0,200,0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imwrite_unicode(output_path, annotated)

    out_frames = [{"x": int(x), "y": int(y), "width": int(w), "height": int(h)} for (x,y,w,h) in frames]
    out_resources = [{"id": i, "x": int(x), "y": int(y), "width": int(w), "height": int(h)} for i,(x,y,w,h) in enumerate(resources)]
    out_yellow = [{"id": i, "x": int(x), "y": int(y), "width": int(w), "height": int(h)} for i,(x,y,w,h) in enumerate(yellow_targets)]
    out_edges_y = [{"from": int(fr), "to": int(to)} for (fr, to) in edges_yellow]

    debug_heads = [
        {"x": float(tip[0]), "y": float(tip[1]), "dx": float(d[0]), "dy": float(d[1]),
         "to": (int(to_idx) if to_idx is not None else None)}
        for (tip, d, to_idx) in arrow_heads_tagged
    ]

    data = {
        "imageWidth": int(W),
        "imageHeight": int(H),
        "frames": out_frames,
        "resources": out_resources,
        "yellowResources": out_yellow,
        "edgesYellow": out_edges_y,
        "edgesPolyline": edges_polyline,
        "debug": {"arrowHeads": debug_heads,
                  "counts": {"heads": len(arrow_heads_tagged),
                             "edges": len(edges_yellow),
                             "polylines": len(edges_polyline)}}
    }
    return json.dumps(data, ensure_ascii=False, indent=2)
