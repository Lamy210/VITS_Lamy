FROM python:3.6

USER root

RUN apt-get update && \
    apt-get install -y locales

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

RUN localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# 分割した RUN コマンド
RUN apt-get install -y vim less
RUN apt install build-essential
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install poetry

# 分割した RUN コマンド
RUN pip install Cython librosa matplotlib numba numpy Pillow scikit-learn scipy six SoundFile
RUN pip install torch torchvision torchaudio

# 事前ビルド済みホイールを使用する
# RUN pip install pyopenjtalk

# ビルドを試す
RUN apt-get install -y build-essential libasound2-dev
RUN pip install pyopenjtalk --no-binary pyopenjtalk
WORKDIR /VITS