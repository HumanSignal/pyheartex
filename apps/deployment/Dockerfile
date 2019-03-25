FROM python:3.7-alpine
RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --no-cache --update-cache add \
        gfortran python python-dev py-pip build-base wget freetype-dev libpng-dev openblas-dev git
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
COPY requirements.txt /tmp/
RUN pip install -U pip
RUN pip install -r /tmp/requirements.txt
