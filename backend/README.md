# cloud-infra-scanner-backend
클라우드 인프라 스캐너 (OpenCV)

![cloudinfra_largepix](https://github.com/user-attachments/assets/696eaaa4-2c95-4178-94db-bc0fbeefaba2)

### **이미지 처리 스크립트 문서화**  

이 문서는 OpenCV와 Tesseract OCR을 활용하여 **이미지를 분석하고 처리하는 Python 스크립트**에 대한 설명을 정리한 것입니다.  

---

## **1. 개요**
이 스크립트는 **이미지에서 특정 객체(프레임 및 자원)를 감지하고, OCR을 사용해 텍스트를 추출한 후, 결과를 시각적으로 출력하는 기능**을 수행합니다.  

- **주요 기능**
  - OpenCV(`cv2`)를 이용한 이미지 처리
  - Tesseract OCR(`pytesseract`)을 이용한 텍스트 인식
  - NumPy(`numpy`)를 이용한 수학적 연산
  - JSON(`json`)을 이용한 결과 데이터 구조화

---

## **2. 함수별 기능 정리**

### **2.1 Tesseract OCR 초기화**
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```
- Tesseract OCR 실행 경로를 설정합니다.  
- Tesseract를 기본 경로에 설치하지 않았다면, 설치된 위치를 반영하여 수정해야 합니다.

---

### **2.2 `merge_similar_frames(frames, threshold=50)`**
```python
def merge_similar_frames(frames, threshold=50):
```
#### **기능**  
- 감지된 프레임(객체)이 비슷한 위치와 크기를 가질 경우, 하나로 병합합니다.  
- 중복된 프레임을 제거하여 최종 결과를 더 간결하게 만듭니다.

#### **동작 방식**
1. 입력된 `frames` 리스트를 순회하며 프레임 간 거리 및 크기 비교  
2. 거리(`distance`), 가로(`width`), 세로(`height`) 차이가 `threshold` 이하라면 병합  
3. 최종적으로 중복이 제거된 프레임 리스트를 반환  

---

### **2.3 `merge_similar_resources(resources, threshold=50)`**
```python
def merge_similar_resources(resources, threshold=50):
```
#### **기능**  
- 이미지 내에서 감지된 자원(Resource) 들이 비슷한 위치와 크기를 가질 경우, 하나로 병합  
- 프레임 병합(`merge_similar_frames`)과 동일한 방식으로 동작

---

### **2.4 `validate_string(data)`**
```python
def validate_string(data):
    return re.sub(r"[^a-zA-Z0-9]", "", data)
```
#### **기능**  
- OCR을 통해 추출된 텍스트에서 영어와 숫자만 남기고 나머지를 제거  
- 특수 문자나 불필요한 공백 등을 정리하여 깨끗한 문자열을 반환  

---

### **2.5 `extract_text_below(image, resource, padding=10)`**
```python
def extract_text_below(image, resource, padding=10):
```
#### **기능**  
- 특정 자원(Resource) 아래 영역에서 텍스트를 OCR로 추출  
- 예를 들어, 버튼 아래에 있는 라벨 텍스트를 감지할 때 유용  

#### **동작 방식**
1. 자원의 위치 `(x, y, w, h)`를 기반으로 아래쪽 `30px` 정도의 영역(ROI) 추출  
2. 해당 영역을 Tesseract OCR을 사용하여 텍스트 변환  
3. 최종적으로 텍스트를 정리하여 반환

---

### **2.6 `process_image(input_path, output_path)`**
```python
def process_image(input_path, output_path):
```
#### **기능**  
- 입력된 이미지를 OpenCV를 활용하여 분석 및 가공  
- 감지된 프레임 및 자원을 표시하고, OCR을 수행하여 텍스트를 추출  
- 최종 결과 이미지(`output_path`)와 JSON 데이터 반환

#### **동작 방식**
1. **이미지 로드**
   ```python
   image = cv2.imread(input_path)
   ```
   - 입력된 이미지를 읽어옴
   - 만약 이미지를 읽을 수 없다면 오류 발생

2. **이미지 전처리**
   ```python
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   blur = cv2.GaussianBlur(gray, (5, 5), 0)
   edges = cv2.Canny(blur, 50, 150)
   ```
   - 그레이스케일 변환 → 블러링 → 엣지 감지 순서로 이미지 전처리

3. **컨투어(객체) 감지**
   ```python
   contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   ```
   - 이미지에서 윤곽선을 찾아 객체의 경계를 감지

4. **프레임 및 자원 분류**
   ```python
   if len(approx) == 4 and cv2.isContourConvex(approx):
   ```
   - 프레임: 네모 형태(`4개 꼭짓점 & 볼록한 컨투어`)를 만족하면 프레임 객체로 저장  
   - 자원: 크기가 일정 범위(`20px ~ 100px`) 내에 있고 원형 또는 다각형이면 자원으로 저장  

5. **병합 및 텍스트 추출**
   ```python
   frames = merge_similar_frames(frames)
   resources = merge_similar_resources(resources)
   ```
   - 중복된 프레임 및 자원을 병합  
   - OCR을 사용하여 각 자원 아래의 텍스트 추출  

6. **결과 이미지 저장**
   ```python
   cv2.imwrite(output_path, annotated_image)
   ```
   - 감지된 프레임과 OCR 텍스트를 포함한 새로운 이미지 파일 저장

7. **JSON 결과 반환**
   ```python
   return json.dumps(data, indent=4)
   ```
   - 최종 감지된 프레임 및 OCR 데이터를 JSON 형식으로 반환  
   - `frames`, `resources` 정보가 포함됨

---

## **3. 최종 결과 예시**
![cloudinfra](https://github.com/user-attachments/assets/d99775d5-e970-4fb3-af6e-001210b53f2d)


### **3.1 OCR을 포함한 JSON 결과 예시**
```json
{
  "frames": [
    {
      "x": "100",
      "y": "200",
      "width": "150",
      "height": "100"
    }
  ],
  "resources": [
    {
      "x": "300",
      "y": "400",
      "width": "50",
      "height": "50",
      "text": "START"
    }
  ]
}
```
- 감지된 프레임과 자원의 위치 및 크기, OCR을 통해 추출된 텍스트를 포함

---

### **3.2 처리된 이미지 출력 예시**
- 출력 이미지에는 다음과 같은 정보가 표시됨  
  - **초록색 박스**: 감지된 프레임  
  - **핑크색 박스 + OCR 텍스트**: 감지된 자원 및 텍스트  

---

## **4. 결론**
- 이미지에서 특정 영역을 감지하고 텍스트를 추출하는 자동화된 스크립트  
- OpenCV를 활용한 프레임/자원 탐지 및 병합 기능 포함  
- Tesseract OCR을 통해 중요한 정보 추출 가능  
- JSON 데이터를 활용하여 프로그램에서 추가적인 처리 가능

---

## **회고**

예전에도 OpenCV를 다뤄본 적이 있어서 내가 익숙한 방법으로 나름의 문제를 해결했지만 역시, OpenCV는 연구가 필요한 분야같다. 연습보다 연구에 가까운 학문적 분야하는 생각.
아쉬운 점은 선의 연결인데, 이걸 중점적으로 하는 프로젝트는 추후 ai등의 연결과 더불어 진행해볼 생각.


