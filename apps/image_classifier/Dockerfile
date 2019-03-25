FROM niklub/pyheartex:apps
ADD . /app
WORKDIR /app
EXPOSE 80
RUN apk add jpeg-dev zlib-dev freetype-dev lcms2-dev openjpeg-dev tiff-dev tk-dev tcl-dev libc6-compat
RUN pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1.post2-cp37-cp37m-linux_x86_64.whl && \
    pip3 install torchvision
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