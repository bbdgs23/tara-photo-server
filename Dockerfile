FROM python:3.9-slim

WORKDIR /app

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY main.py .

# 포트 노출 (5050으로 변경)
EXPOSE 5050

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5050"]