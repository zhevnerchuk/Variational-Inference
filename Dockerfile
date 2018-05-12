# Base image
FROM python:3.6

# Author information
MAINTAINER zhevnerchuk@gmail.com

# Set a working directory
WORKDIR home/

# Install necessary libraries
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision
RUN pip3 install numpy matplotlib pyro-ppl

RUN git clone https://github.com/zhevnerchuk/Variational-Inference

CMD bash
