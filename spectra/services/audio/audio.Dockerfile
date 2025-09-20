FROM python:3.11-slim-buster AS base

# set environment
ENV WD="/root/spectra"
ENV DEBIAN_FRONTEND=noninteractive
USER 0
WORKDIR $WD