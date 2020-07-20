FROM python:3.7
LABEL maintainer "Henry Lloyd-Laney henry.lloyd-laney@cs.ox.ac.uk"

COPY ./ ./
WORKDIR ./code
RUN apt-get update && apt-get -y install cmake && apt-get -y install libboost-math-dev
RUN  cmake .
RUN make

RUN pip3 install -r /requirements.txt
EXPOSE 8050
CMD ["python", "./app.py"]
