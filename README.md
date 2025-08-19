# Cloud Infra Scanner

클라우드 인프라 스캐너 (Next.js, TypeScript, Python, OpenCV)
클라우드 인프라 스캐너는 클라우드 아키텍처 다이어그램을 자동으로 인식하여 각 리소스를 적절한 위치에 배치하고 연결해주는 프로젝트입니다.

##주요 기능

업로드된 아키텍처 다이어그램 이미지 인식 (OpenCV, Python)

클라우드 리소스 자동 탐지 및 분류

리소스별 위치 매핑 및 선 연결 처리

결과물을 시각적으로 재가공하여 다운로드 가능

##기술 스택

Frontend: Next.js, TypeScript, React

Backend: Python (Flask), OpenCV, JSON

Infra/Etc: Git, GitHub Actions, Docker (선택적)
---

## 📂 프로젝트 구조

```
cloud-infra-scanner/
├── backend/            # Python + Flask 서버
│   ├── app/
│   │   ├── process_image.py   # OpenCV 이미지 처리 로직
│   │   ├── server.py          # Flask 엔드포인트 정의
│   │   ├── uploads/           # 업로드된 원본 이미지
│   │   └── outputs/           # 처리된 결과 이미지
│   └── requirements.txt       # Python 의존성 목록
│
├── frontend/           # Next.js + TypeScript 클라이언트
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── tsconfig.json
│
├── outputs/            # 저장된 최종 결과물
├── uploads/            # 업로드된 리소스 (공유용)
├── cd/                 # 배포 관련 설정 (예: Docker, CI/CD)
├── python/             # Python 유틸 스크립트 모음
└── README.md           # 프로젝트 설명 문서
```

---

## 🚀 실행 방법

### Backend (Flask)

```bash
cd backend/app
python server.py
```

* 기본 실행: [http://127.0.0.1:5000](http://127.0.0.1:5000)
* 주요 엔드포인트:

  * `POST /process-image` → 이미지 업로드 및 처리
  * `GET /outputs/<filename>` → 처리된 이미지 반환

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

* 기본 실행: [http://localhost:3000](http://localhost:3000)

---

## ⚙️ 환경 변수 예시

* `.env` (backend)

```env
FLASK_ENV=development
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
```

* `.env.local` (frontend)

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:5000
```

---

## 🌿 Git Workflow

* `main`: 안정화된 배포용 브랜치
* `develop`: 개발 통합 브랜치
* `feature/*`: 기능별 작업 브랜치

예시:

```bash
# 새로운 기능 작업
git checkout -b feature/image-connector develop

# 작업 완료 후
git commit -m "[Feat]: 이미지 연결 기능 추가"
git push origin feature/image-connector

# PR → develop → main 순으로 머지
```

---

## 🧪 기술 스택

* **Frontend**: Next.js, React, TypeScript
* **Backend**: Python, Flask, OpenCV
* **Infra**: Docker, GitHub Actions (CI/CD 예정)

---

## 📌 TODO

* [ ] OpenCV 이미지 분석 정확도 개선
* [ ] 프론트엔드 UI 개선 (노드/선 시각화)
* [ ] 배포 자동화 (Docker + GitHub Actions)
* [ ] README 영문 버전 작성

---

## 👩‍💻 기여 방법

1. 이슈 생성 후 작업 브랜치 생성
2. 커밋 컨벤션: `[Feat]: …`, `[Fix]: …`, `[Refactor]: …`
3. PR 생성 후 코드 리뷰 요청

---

## 📜 라이선스

MIT License


<img width="1866" height="1295" alt="image" src="https://github.com/user-attachments/assets/8c5291e9-1e1e-4727-a7fc-f26e89b8de94" />


<img width="1682" height="1297" alt="image" src="https://github.com/user-attachments/assets/a5e891e9-5369-4e59-8e21-1c567c475568" />
