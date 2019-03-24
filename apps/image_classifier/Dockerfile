FROM python:3.7-alpine
ADD . /app
WORKDIR /app
EXPOSE 80
RUN pip install -r requirements.txt
CMD ["python", "image_classifier.py"]