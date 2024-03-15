FROM python:slim

RUN pip install holoviews panel hvplot antspyx
RUN apt update
RUN apt-get install git -y
RUN git clone https://github.com/fcomitani/simpsom
RUN cd simpsom && python setup.py install

RUN git clone https://github.com/ahsanMah/sade
RUN cd sade && python -m pip install .

RUN pip install tabulate