# hash:sha256:7e424c1ba686bf22cd25eeef785d779b2c5eafc5bbf9c222165ab06bffd5a452
FROM registry.codeocean.com/codeocean/miniconda3:4.5.11-python3.7-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        conda==4.13.0 \
    && conda clean -ya
RUN conda install -y \
        jupyter==1.0.0 \
        tensorflow==1.14.0 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    keras==2.1.6 \
    matplotlib==3.5.3 \
    numpy==1.21.6 \
    sklearn==0.0
