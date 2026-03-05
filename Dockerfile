# Python 3.9 이미지 기반
FROM python:3.9-slim

# 시스템 라이브러리 설치 (Azure Speech SDK 필수)
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libasound2 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 포트 개방 및 서버 실행
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]