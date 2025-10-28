FROM python

COPY ./pta /project/pta
COPY ./main.py /project

WORKDIR /project

VOLUME /project/data

RUN pip install tqdm typer numpy matplotlib typer

ENTRYPOINT ["python", "main.py"]