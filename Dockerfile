FROM python:3.11-slim-buster AS base

# set environment
ENV WD="/root/spectra"
ENV DEBIAN_FRONTEND=noninteractive
USER 0
WORKDIR $WD

FROM base AS install_dependencies
# install dependencies
RUN apt-get update -y && apt install -y cmake git libglew-dev libpango1.0-dev pkg-config nasm

# install ffmpeg
WORKDIR $WD/bin
RUN git clone https://github.com/FFmpeg/FFmpeg.git
WORKDIR $WD/bin/FFmpeg
RUN $WD/bin/FFmpeg/configure
RUN make -j4
RUN make install
WORKDIR $WD

FROM install_dependencies AS python_dependencies
# conda install
COPY --from=continuumio/miniconda3 /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
RUN set -ex && \
    conda config --set always_yes yes --set changeps1 no && \
    conda info -a && \
    conda config --add channels conda-forge && \
    conda install --quiet --freeze-installed -c main conda-pack pip python=3.11

COPY ./requirements.txt $WD/requirements.txt
RUN pip3 install -r $WD/requirements.txt

FROM python_dependencies AS hot_reload
COPY . $WD/

ENTRYPOINT [ "bash" ]