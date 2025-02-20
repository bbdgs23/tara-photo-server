FROM python:3.10-slim

WORKDIR /app

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p uploads outputs temp

# 포트 노출
EXPOSE 5050

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 실행 명령어
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]