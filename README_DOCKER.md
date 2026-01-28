# Docker Quickstart

This document explains how to build, load, and run the Docker environment for this project.  
It is intentionally separate from the main `README.md` and only covers Docker usage, common workflows, and image sharing.
---
## 1. Install and Enable Docker

Follow the official Docker installation guide for your platform:

https://docs.docker.com/engine/install/

Make sure Docker is running before proceeding.

---

## 2. Use a Prebuilt Docker Image
If you have a prebuilt image archive (`dafny_research_latest.tar`), load it with:

```shell
docker load -i dafny_research_latest.tar
``` 
To save the current image to a file
```shell 
docker save -o dafny_research_latest.tar dafny_research:latest
``` 

## 3. Build the docker image (from repository Root)
This repository includes a ready-to-use Dockerfile. 
```shell 
docker build -t dafny_research:latest .
```

## 4. Run an Interactive Container 
Run an interactive container (forwarding port 8888 to use jupyter notebook):
```shell
docker run --rm -it \
  -p 8888:8888 \
  -w /app \
  dafny_research:latest bash
```

## 5.Passing Environment Variables (e.g. OpenAI API Key)
```shell
docker run --rm -it \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -w /app \
  dafny_research:latest bash
``` 

## 6. Development Mode (Mount Local Source Directory)
```shell
docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)/src:/app/src:delegated" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -w /app \
  dafny_research:latest bash
``` 

## 7. Running Jupyter notebooks (To see the daya analyses sections)
Jupyter is available inside the container.
After starting the container with port forwarding (-p 8888:8888), run in the browser:
```shell
jupyter lab --ip=0.0.0.0 --no-browser
``` 
You will get a line as ouput saying which links to copy to acess the file in the browser:
For instances (for one run the following output was obtained):
```txt
    To access the server, open this file in a browser:
        file:///home/researcher/.local/share/jupyter/runtime/jpserver-12-open.html
    Or copy and paste one of these URLs:
        http://42807fdb1356:8888/lab?token=bc6ea10c52c351112275cf9c05bfd7dd9bfe376bb569554b
        http://127.0.0.1:8888/lab?token=bc6ea10c52c351112275cf9c05bfd7dd9bfe376bb569554b

```

Copy the final url , in the case above:
```txt
       http://127.0.0.1:8888/lab?token=bc6ea10c52c351112275cf9c05bfd7dd9bfe376bb569554b
``` 
In your browser on the environment outside docker and you will be have to see all files on the docker solution. Navigate there to the jupyter notebooks


## 8. Insede the container 
Once inside the container, follow the instructions in the main project documentation (README.md)
