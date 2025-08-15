import cv2
import numpy as np
import pytesseract
import os
import json
import re

# Tesseract 경로 설정 (필요 시 변경)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def merge_similar_frames(frames, threshold=50):
    """
    비슷한 크기와 위치의 틀을 하나로 병합.
    """
    merged = []
    for frame in frames:
        x, y, w, h = frame
        merged_flag = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            cx1, cy1 = x + w / 2, y + h / 2
            cx2, cy2 = mx + mw / 2, my + mh / 2
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if distance < threshold and abs(w - mw) < threshold and abs(h - mh) < threshold:
                x_min = min(x, mx)
                y_min = min(y, my)
                x_max = max(x + w, mx + mw)
                y_max = max(y + h, my + mh)
                merged[i] = (x_min, y_min, x_max - x_min, y_max - y_min)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(frame)
    return merged

def merge_similar_resources(resources, threshold=50):
    """
    비슷한 크기와 위치의 자원을 하나로 병합.
    """
    merged = []
    for resource in resources:
        x, y, w, h = resource
        merged_flag = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            cx1, cy1 = x + w / 2, y + h / 2
            cx2, cy2 = mx + mw / 2, my + mh / 2
            distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if distance < threshold and abs(w - mw) < threshold and abs(h - mh) < threshold:
                x_min = min(x, mx)
                y_min = min(y, my)
                x_max = max(x + w, mx + mw)
                y_max = max(y + h, my + mh)
                merged[i] = (x_min, y_min, x_max - x_min, y_max - y_min)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(resource)
    return merged

def validate_string(data):
    """
    데이터에서 영어와 숫자만 포함하도록 필터링.
    """
    valid_data = re.sub(r"[^a-zA-Z0-9]", "", data)
    return valid_data

def extract_text_below(image, resource, padding=10):
    """
    자원 바로 아래 텍스트를 OCR로 추출.
    """
    x, y, w, h = resource
    roi = image[y + h + padding : y + h + padding + 30, x : x + w]  # 아래 30px 영역
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        text = pytesseract.image_to_string(roi, lang='eng', config='--psm 6')
        return text.strip()
    return ""

def process_image(input_path, output_path):
    """
    OpenCV를 사용하여 이미지를 처리하고 결과를 저장.
    """
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image from {input_path}. Please check if the file exists and is a valid image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    frames = []
    resources = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            aspect_ratio = max(w / h, h / w)
            if w > 100 and h > 100 and area > 1000 and aspect_ratio <= 2:
                frames.append((x, y, w, h))
                continue

        if 20 < w < 100 and 20 < h < 100 and area > 50:
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio <= 2:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity > 0.7 or len(approx) > 5:
                    resources.append((x, y, w, h))

    frames = merge_similar_frames(frames)
    resources = merge_similar_resources(resources)

    resource_texts = []
    for resource in resources:
        text = extract_text_below(image, resource)
        resource_texts.append({"resource": resource, "text": validate_string(text)})

    annotated_image = image.copy()
    for fx, fy, fw, fh in frames:
        cv2.rectangle(annotated_image, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)

    for item in resource_texts:
        rx, ry, rw, rh = item["resource"]
        text = item["text"]
        cv2.rectangle(annotated_image, (rx, ry), (rx + rw, ry + rh), (255, 0, 255), 2)
        cv2.putText(annotated_image, text, (rx, ry + rh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, annotated_image)

    data = {
        "frames": [{"x": validate_string(str(fx)), "y": validate_string(str(fy)), "width": validate_string(str(fw)), "height": validate_string(str(fh))} for fx, fy, fw, fh in frames],
 "resources": [
        {
            "x": validate_string(str(item["resource"][0])),  # x
            "y": validate_string(str(item["resource"][1])),  # y
            "width": validate_string(str(item["resource"][2])),  # width
            "height": validate_string(str(item["resource"][3])),  # height
            "text": validate_string(item["text"])  # text
        }
        for item in resource_texts
    ]    }
    return json.dumps(data, indent=4)
