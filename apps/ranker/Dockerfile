FROM niklub/pyheartex:apps
ADD . /app
WORKDIR /app
EXPOSE 80
RUN pip install -r requirements.txt
RUN mkdir models
CMD ["python", "ranker.py", \
    "--model-dir=models", \
    "--update-period=10", \
    "--min-examples=10", \
    "--port=80" \
]