# cloud-infra-scanner
클라우드 인프라 스캐너 (Nextjs, Typescript, Python, OpenCV)

📂 프로젝트 구조
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
└── README.md           # 프로젝트 설명 

<img width="1866" height="1295" alt="image" src="https://github.com/user-attachments/assets/8c5291e9-1e1e-4727-a7fc-f26e89b8de94" />


<img width="1682" height="1297" alt="image" src="https://github.com/user-attachments/assets/a5e891e9-5369-4e59-8e21-1c567c475568" />
