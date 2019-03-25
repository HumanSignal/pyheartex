FROM niklub/pyheartex:apps
ADD . /app
WORKDIR /app
EXPOSE 80
RUN pip install -r requirements.txt
RUN mkdir models
CMD ["python", "sample_app.py", \
    "--model-dir=models", \
    "--from-name=sample_class", \
    "--to-name=sample_class", \
    "--data-field=sample", \
    "--update-period=10", \
    "--min-examples=10", \
    "--port=80" \
]
