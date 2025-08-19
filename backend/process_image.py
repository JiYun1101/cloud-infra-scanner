# # process_image.py
# import os
# import re
# import json
# import cv2
# import numpy as np
# import pytesseract

# # --- Tesseract 경로 (필요시 .env에서 덮어쓰기) ---
# # 예: TESSERACT_CMD="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
# TESSERACT_CMD = os.getenv(
#     "TESSERACT_CMD",
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# # ---------- 유틸 ----------
# def imread_unicode(path: str):
#     """한글/특수문자 경로에서도 안전하게 이미지 읽기"""
#     stream = np.fromfile(path, dtype=np.uint8)
#     return cv2.imdecode(stream, cv2.IMREAD_COLOR)

# def _validate_label(text: str) -> str:
#     """라벨 텍스트 정리(영문/숫자만, 너무 짧으면 노이즈로 간주)"""
#     if not text:
#         return ""
#     t = re.sub(r"[^A-Za-z0-9]", "", text)
#     return t if len(t) >= 2 else ""

# def _merge_similar(rects, threshold=40):
#     """가까운 박스들을 병합(빠른 근사 병합)"""
#     if not rects:
#         return []
#     rects = sorted(rects, key=lambda r: (r[0], r[1]))
#     merged = []
#     for (x, y, w, h) in rects:
#         if not merged:
#             merged.append([x, y, w, h])
#             continue
#         X, Y, W, H = merged[-1]
#         if (abs(x - X) <= threshold and abs(y - Y) <= threshold and
#             abs(w - W) <= threshold and abs(h - H) <= threshold):
#             nx = min(x, X); ny = min(y, Y)
#             nx2 = max(x + w, X + W); ny2 = max(y + h, Y + H)
#             merged[-1] = [nx, ny, nx2 - nx, ny2 - ny]
#         else:
#             merged.append([x, y, w, h])
#     return [tuple(m) for m in merged]

# def _nms(rects, iou_thresh=0.3):
#     """비최대 억제(NMS)로 중복 박스 제거"""
#     if not rects:
#         return []
#     boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=np.float32)
#     scores = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # 면적을 점수로
#     order = scores.argsort()[::-1]
#     keep = []

#     while order.size > 0:
#         i = order[0]
#         keep.append(i)

#         xx1 = np.maximum(boxes[i, 0], boxes[order, 0])
#         yy1 = np.maximum(boxes[i, 1], boxes[order, 1])
#         xx2 = np.minimum(boxes[i, 2], boxes[order, 2])
#         yy2 = np.minimum(boxes[i, 3], boxes[order, 3])

#         w = np.maximum(0, xx2 - xx1)
#         h = np.maximum(0, yy2 - yy1)
#         inter = w * h

#         area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
#         area_j = (boxes[order, 2] - boxes[order, 0]) * (boxes[order, 3] - boxes[order, 1])
#         iou = inter / (area_i + area_j - inter + 1e-6)

#         order = order[1:][iou[1:] < iou_thresh]

#     return [rects[i] for i in keep]

# def _extract_text_below(image, rect, padding=8, below=32):
#     """박스 바로 아래 라벨을 OCR로 추출"""
#     x, y, w, h = rect
#     H, W = image.shape[:2]
#     y1 = min(H, y + h + padding)
#     y2 = min(H, y + h + padding + below)
#     x1 = max(0, x)
#     x2 = min(W, x + w)
#     roi = image[y1:y2, x1:x2]
#     if roi.size == 0:
#         return ""

#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#     th = cv2.dilate(th, np.ones((1, 2), np.uint8), 1)  # 가로로 약간 두껍게

#     txt = pytesseract.image_to_string(
#         th,
#         config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
#     )
#     return _validate_label(txt)


# # ---------- 메인 처리 ----------
# def process_image(input_path: str, output_path: str) -> str:
#     """
#     이미지를 분석해 프레임/리소스를 감지하고 주석 이미지를 저장.
#     결과 메타데이터(JSON 문자열)를 반환합니다.
#     반환 JSON에는 imageWidth/imageHeight가 포함됩니다(프론트 스케일 맞춤용).
#     """
#     input_path = os.path.abspath(input_path)
#     output_path = os.path.abspath(output_path)

#     image = imread_unicode(input_path)
#     if image is None:
#         raise ValueError(
#             f"Cannot read image from {input_path}. Please check if the file exists and is a valid image."
#         )

#     # --- 전처리 ---
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#     edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

#     # --- 컨투어 추출 ---
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     frames, resources = [], []
#     for cnt in contours:
#         peri = cv2.arcLength(cnt, True)
#         if peri == 0:
#             continue
#         approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#         x, y, w, h = cv2.boundingRect(approx)
#         area = cv2.contourArea(cnt)
#         if w * h < 500 or w < 15 or h < 15:  # 너무 작은 후보 제거
#             continue

#         # 프레임(큰 사각형, 비교적 반듯함)
#         if len(approx) == 4 and cv2.isContourConvex(approx):
#             if area > 2000 and max(w / h, h / w) <= 2.0:
#                 frames.append((x, y, w, h))
#                 continue

#         # 리소스(중간 크기, 원형/다각형 포함)
#         if 25 < w < 180 and 25 < h < 180 and area > 120:
#             circularity = 4 * np.pi * area / (peri ** 2) if peri else 0
#             if circularity > 0.55 or len(approx) >= 5:
#                 resources.append((x, y, w, h))

#     # --- 병합 + NMS로 중복 억제 ---
#     frames = _nms(_merge_similar(frames, threshold=40), iou_thresh=0.20)
#     resources = _nms(_merge_similar(resources, threshold=25), iou_thresh=0.30)

#     # --- 주석 이미지 생성 ---
#     annotated = image.copy()
#     out_frames = []
#     out_resources = []

#     # 초록: 프레임
#     for (x, y, w, h) in frames:
#         cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
#         out_frames.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})

#     # 보라: 리소스 + 라벨
#     for (x, y, w, h) in resources:
#         label = _extract_text_below(image, (x, y, w, h))
#         cv2.rectangle(annotated, (x, y), (x + w, y + h), (180, 0, 180), 2)
#         if label:
#             cv2.putText(
#                 annotated, label, (x + w + 6, y + h // 2),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 0, 180), 2, cv2.LINE_AA
#             )
#         out_resources.append({
#             "x": int(x), "y": int(y), "width": int(w), "height": int(h), "text": label
#         })

#     # 저장
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     cv2.imwrite(output_path, annotated)

#     # 프론트 스케일 동기화를 위해 원본 크기 포함
#     H, W = image.shape[:2]
#     data = {
#         "imageWidth": int(W),
#         "imageHeight": int(H),
#         "frames": out_frames,
#         "resources": out_resources
#     }
#     return json.dumps(data, ensure_ascii=False, indent=2)
