# Dockerfile (UE packaged binary is NOT baked in; mount it to /Linux at runtime)
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV VK_LAYER_PATH=""
ENV VK_INSTANCE_LAYERS=""

ENV UE_PORT=9000
ENV JUPYTER_PORT=8888
ENV DISPLAY=:99
ENV WORKDIR=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    ca-certificates locales tzdata \
    libstdc++6 \
    libglib2.0-0 \
    libx11-6 libxext6 libxrender1 libxcb1 \
    libxi6 libxrandr2 libxinerama1 libxcursor1 libxss1 libxkbcommon0 \
    libxtst6 libxv1 libegl1 libglu1-mesa \
    libvulkan1 vulkan-tools libvulkan-dev \
    xvfb xauth x11-xserver-utils \
    curl git unzip procps gosu netcat lsof \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user ONLY for running UE (Unreal disallows admin/root)
ARG USERNAME=sim
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

COPY SimWorldGym /SimWorldGym

RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install --no-cache-dir notebook \
 && python3 -m pip install --no-cache-dir -e /SimWorldGym

RUN mkdir -p /tmp/.X11-unix && chmod 1777 /tmp/.X11-unix

COPY entrypoint.sh /entrypoint.sh
COPY userscript.sh /userscript.sh
RUN chmod +x /entrypoint.sh /userscript.sh

ENTRYPOINT ["/entrypoint.sh"]