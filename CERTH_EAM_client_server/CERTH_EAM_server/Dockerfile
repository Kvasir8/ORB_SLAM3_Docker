ARG distro="18.04"
FROM ubuntu:${distro}

ENV TZ=Europe/Athens
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install basics
RUN apt-get -y update \
 && apt-get -y install --no-install-recommends apt-utils software-properties-common \
 && apt-add-repository -y ppa:deadsnakes/ppa -u

RUN apt-get -y install --no-install-recommends wget git ca-certificates bzip2 cmake vim \
  && apt-get -y install --no-install-recommends libsm6 libglib2.0-0 ffmpeg \
  && apt-get -y install --no-install-recommends libxext6 libxrender-dev virtualenv python3.6-dev g++\
  && apt-get -y install python3-pip

# Setup Virtualenv
RUN python3 -m virtualenv --python=/usr/bin/python3.6 /opt/venv
RUN . /opt/venv/bin/activate
RUN pip3 install --upgrade pip
RUN pip3 install numpy
RUN pip3 install opencv-contrib-python

COPY certh_server.py .
COPY frame_message.py .

RUN rm -rf /var/lib/apt/lists/*