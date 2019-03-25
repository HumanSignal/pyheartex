FROM python:3.7
RUN pip install uwsgi
ADD . /app
WORKDIR /app
EXPOSE 10001
RUN mkdir models && mkdir images
RUN pip install uwsgi
CMD ["uwsgi", "--ini", "wsgi.ini"]