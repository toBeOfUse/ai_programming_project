FROM python:3.9.15-slim

WORKDIR /ai-PHW

RUN apt-get update && apt install -y python3-tk

RUN pip install numpy && pip install matplotlib && pip install seaborn

COPY . .

COPY matplotlib /root/.config/matplotlib/

ARG export MATPLOTLIBRC=/root/.config/matplotlib/

CMD [ "sleep", "infinity"]