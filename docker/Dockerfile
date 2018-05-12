# Base image
FROM python:3.6

# Author information
MAINTAINER zhevnerchuk@gmail.com

# Set a working directory
WORKDIR home/

RUN apt-get update
# Install necessary libraries
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision
RUN pip3 install numpy matplotlib pyro-ppl

RUN pip3 install conda
RUN pip3 install ipykernel jupyter

RUN git clone https://github.com/zhevnerchuk/Variational-Inference

# ------ Jupyter ------
# Install python3 kernel
RUN python3 -m ipykernel.kernelspec

# Setup extensions
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

# Set up notebook config
COPY ./build_sources/jupyter_notebook_config.py /root/.jupyter/
COPY ./build_sources/run_jupyter.sh /
RUN chmod +x /run_jupyter.sh


# ------ Docker setup ------
# Jupyter notebook
EXPOSE 8888

CMD jupyter notebook --allow-root
