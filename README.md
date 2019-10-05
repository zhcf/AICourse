**- Prepare a server with GPU and install OS.** 

**- Install nvidia-dirver** 

**- Install docker-ce** 

**- Install nvidia-docker plugin  https://github.com/NVIDIA/nvidia-docker**

**- Git clone this repository to host directory /AICourse**

**- Download data and env need by this course**

```
Data and env are saved on baidu yunpan: https://pan.baidu.com/s/1FbODrDjFJf1IE7iFdl9GuQ extract code: p96b

The data data.tar.parta* used by this course is almost 10GB, download data.tar.parta* to /AICourse, then: 
cat data.tar.parta* > data.tar
tar -xf data.tar
rm -rf data.tar.parta*

The env(library) env.tar used by this course is almost 4GB, download env.tar to /AICourse, then: 
tar -xf env.tar
```

**- Launch jupyter and use jupyter for running the course program.**

```
docker run -it --gpus all -p 8080:8888 -v /AICourse:/AICourse  zhcf/aicourse:20191004 
cd /AICourse/env 
source bin/activate 
jupyter notebook --no-browser --allow-root --NotebookApp.ip='0.0.0.0' --NotebookApp.port=8888  --NotebookApp.allow_origin='*' --notebook-dir=/AICourse
```
