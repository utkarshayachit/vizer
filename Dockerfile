FROM ubuntu:22.04
EXPOSE 8080

# install pre-requisites
RUN apt update && \
    apt install -y \
        build-essential \
        cmake \
        git \
        libcurl4 \
        libglapi-mesa \
        libgomp1 \
        ninja-build \
        software-properties-common \
        wget && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    DEBIAN_FRONTEND=noninteractive apt install python3.9 python3.9-venv python3.9-distutils -y && \
    apt -y clean

WORKDIR /opt/paraview
RUN wget https://www.paraview.org/files/v5.10/ParaView-5.10.1-osmesa-MPI-Linux-Python3.9-x86_64.tar.gz && \
    tar xf ParaView-5.10.1-osmesa-MPI-Linux-Python3.9-x86_64.tar.gz --strip-component 1 -C /opt/paraview && \
    rm *.tar.gz

# RUN wget https://www.paraview.org/files/v5.11/ParaView-5.11.0-RC1-osmesa-MPI-Linux-Python3.9-x86_64.tar.gz && \
    # tar xf ParaView-5.11.0-RC1-osmesa-MPI-Linux-Python3.9-x86_64.tar.gz --strip-component 1 -C /opt/paraview && \
    # rm *.tar.gz


WORKDIR /opt/trame
RUN python3.9 -m venv env

COPY pyproject.toml /opt/vizer/
COPY server.py /opt/vizer/
COPY reader.py /opt/vizer/
COPY vizer /opt/vizer/vizer/

WORKDIR /opt/vizer
RUN /bin/bash -c "source /opt/trame/env/bin/activate && python -m pip install --upgrade pip && pip install -e ."

WORKDIR /opt/paraview
ENTRYPOINT ["/opt/paraview/bin/pvpython", \
     "/opt/vizer/server.py", "--server",\
     "--venv", "/opt/trame/env", \
     "-i", "0.0.0.0", "-p", "8080"]