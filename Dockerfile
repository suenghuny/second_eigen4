# 기본 이미지로 Python 3.8을 사용
FROM python:3.8-slim

# 필요한 파일 시스템 도구 설치
RUN apt-get update && apt-get install -y \
    git \
    bash \
    curl

# smac 라이브러리 설치
RUN pip install git+https://github.com/oxwhirl/smac.git

# sacred 설치
RUN pip install sacred

# 프로젝트 디렉토리 설정
WORKDIR /app

# 현재 디렉토리의 파일들을 이미지의 /app 디렉토리로 복사
COPY . /app

# 필요한 라이브러리 설치
RUN bash install_sc2.sh && \
    pip install vessl && \
    pip install numpy==1.22.3 && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install pandas && \
    pip install matplotlib && \
    pip install tensorboard

# 컨테이너 시작 시 실행할 명령어 설정
CMD ["python", "main_rev_agent_grouping.py", "--map_name", "MMM2"]