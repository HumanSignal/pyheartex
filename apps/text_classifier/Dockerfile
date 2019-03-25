FROM python:3.7
RUN pip install uwsgi
ADD . /app
WORKDIR /app
EXPOSE 9090
RUN pip install -r requirements.txt
RUN mkdir models
CMD ["uwsgi", "--ini", "wsgi.ini"]