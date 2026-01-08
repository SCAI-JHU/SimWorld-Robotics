
# SimWorldGym Docker (Unreal Engine + UnrealCV + Jupyter)

This repository provides a Docker-based runtime for running SimWorld (UE 5.3)
with UnrealCV and an optional Jupyter Notebook interface.
It is designed for GPU-accelerated, headless execution using NVIDIA Docker runtime on Linux.

------------------------------------------------------------
## 1. System Requirements

Hardware:
- NVIDIA GPU with Vulkan support

Software:
- Docker
- NVIDIA Container Toolkit
- Docker-level permissions
  
The default base image is:

`nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`

This can be changed in the Dockerfile if a different CUDA or OS version is required.

------------------------------------------------------------
## 2. Directory Layout

Place the following files at the same level as SimWorldGym:

```
<project_root>/
├── Dockerfile
├── .dockerignore
├── entrypoint.sh
├── userscript.sh
├── SimWorldGym/
│   ├── setup.py
└── └── gym_citynav/
```

Notes:

Unreal binaries are NOT baked into the image and should be mounted at runtime

From the project root:

`docker build -t simworld-jupyter:latest .`

After building the image, download and unzip the SimWorld Linux runtime from [SimWorld](https://github.com/SimWorld-AI/SimWorld).

We support both the [base](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Linux-v0_1_0-Foundation.zip) package and the [100-Map](https://simworld-release.s3.us-east-1.amazonaws.com/SimWorld-Linux-v0_1_0-100Maps.zip) version.

------------------------------------------------------------
## 3. Run the Container

General command template:

```
docker run --rm -it \
  --gpus all \
  --runtime=nvidia \
  -p <UnrealCV_port>:<UnrealCV_port> \
  -p <Jupyter_port>:<Jupyter_port> \
  -v <path_to_SimWorldGym>:/SimWorldGym \
  -v <path_to_SimWorld_Linux>:/Linux:rw \
  -v <workspace_dir>:/workspace:rw \
  -e DISPLAY=:<display_number> \
  -e UE_PORT=<UnrealCV_port> \
  -e JUPYTER_PORT=<Jupyter_port> \
  -e UE_MAP=<Map_URL> \
  -e WORKDIR=/workspace \
  simworld-jupyter:latest
```

Required:
- `UE_PORT`
  Port used by UnrealCV (must match -p mapping)

- `JUPYTER_PORT`
  Port for Jupyter Notebook

- `DISPLAY`
  Xvfb display port (e.g. :88), choose randomly while avoiding conflict

- `-v <workspace_dir>:/workspace:rw`
  Controls where logs and code are written.

Optional:
- `-e UE_MAP`
  Example:
  `/Game/TokyoStylizedEnvironment/Maps/Tokyo.umap`
  
  If not provided, the default map is used.
  Only supported by the 100Map build.

------------------------------------------------------------
## 4. Notes and Best Practices

- Always use --runtime=nvidia for Vulkan stability
- Large assets and maps should be mounted, not baked into the image
- Check vulkaninfo by running `!vulkaninfo --summary` in a single cell if connection fails, sometimes changing DISPLAY might help
- Inside the container, Unreal logs can be found at `/workspace/unreal.log`

The environment is now ready for reproducible SimWorld experiments.
