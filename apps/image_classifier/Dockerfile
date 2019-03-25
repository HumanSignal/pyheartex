FROM python:3.7
ADD . /app
WORKDIR /app
EXPOSE 80
RUN mkdir models && mkdir images
CMD ["python", "image_classifier.py", \
    "--model-dir=models", \
    "--image-folder=images", \
    "--from-name=image_class", \
    "--to-name=image_class", \
    "--data-field=image", \
    "--update-period=10", \
    "--min-examples=10", \
    "--port=80" \
]