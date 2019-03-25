FROM niklub/pyheartex:apps
ARG lang
ADD . /app
WORKDIR /app
EXPOSE 80
RUN pip install -r requirements.txt
RUN mkdir models
CMD ["python", "text_classifier.py", \
    "--model-dir=models", \
    "--lang=$lang", \
    "--from-name=text_class", \
    "--to-name=text_class", \
    "--data-field=text", \
    "--update-period=10", \
    "--min-examples=10", \
    "--port=80" \
]