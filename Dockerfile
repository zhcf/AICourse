FROM tensorflow/tensorflow:1.14.0-gpu-py3
 
MAINTAINER zhcf
 
USER root
 
RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install virtualenv