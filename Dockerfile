FROM ubuntu:18.04

RUN apt update

# install MPI
RUN apt install -y openmpi-bin libopenmpi-dev

# install git
RUN apt install -y git

# install cmake make g++
RUN apt install -y cmake make g++

# git clone
RUN git clone https://github.com/kurama622/PicoPebble.git /app/PicoPebble

WORKDIR /app/PicoPebble

RUN ./build_run.sh 3

CMD ["bash"]
