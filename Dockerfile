FROM python:3.12-bookworm

RUN mkdir -p /usr/src/stt_tts
WORKDIR /usr/src/stt_tts
COPY requirements.txt /usr/src/stt_tts/
RUN \
    python3 -m pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt
COPY . .