# merlin-demo

A Demo for [NVIDIA Merlin](https://developer.nvidia.com/merlin) - A framework for training & serving Deep Learning Recommendation Models.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/0*5B_s6eui101ctaHM.png" width="600">

## Prerequisites

1. [NVIDIA GPU](https://developer.nvidia.com/cuda-gpus)

1. Disable secure boot.

   Verify the status of the secure boot with the following command:

   ```
   mokutil --sb-state
   ```

1. Install a supported container engine (Docker, Containerd, CRI-O, Podman).

   - **In this tutorial we will use [Docker](https://docs.docker.com/engine/install).**

     Make sure to [add your user to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/):

     ```
     sudo usermod -aG docker $USER
     ```

     Then restart the machine with:

     ```
     sudo reboot
     ```

1. Install [NVIDIA GPU Drivers](https://www.nvidia.com/en-in/drivers/unix/) + [NVIDIA CUDA Drivers](https://developer.nvidia.com/cuda-downloads) + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) + [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).

   You can use the following installation script for Ubuntu machines:

   https://github.com/taljacob2/NvidiaCUDAForUbuntu/blob/master/install-nvidia-drivers-and-cuda-for-ubuntu.sh

   ```
   sudo bash install-nvidia-drivers-and-cuda-for-ubuntu.sh -s -r -o -d -c=docker
   ```

1. Connect to NVIDIA Registry:

   1. [Create an account](https://ngc.nvidia.com/signup/complete-profile)

      Created an account and log in

   1. [Generate API Key](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key)

      Once you have logged in to the account,

      Go to ["Setup"](https://org.ngc.nvidia.com/setup) -> "Generate Personal Key"

      And generate a new key:

      ![image.png](https://i.imgur.com/5eUwqxH.png)

   1. Login to the NVIDIA Registry:

      ```
      sudo docker login nvcr.io
      ```

      Set the Username to `$oauthtoken`.

      Set the Password to `<YOUR_API_TOKEN>` generated from the previous step.

## Usage

It is preferable to seperate the "training" container from the "serving" container".

### Training Container

```
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
```

> This mutes the non-fatal warnings of:
> ```
> 2025-03-22 12:25:38.494146: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
> ```
> See https://github.com/tensorflow/tensorflow/issues/42738#issuecomment-922422874

> NOTE: We set 8889 as optional port for "feast ui" in [02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb](Merlin/examples/Building-and-deploying-multi-stage-RecSys/02-Deploying-multi-stage-RecSys-with-Merlin-Systems.ipynb)

If you want to use GPU(s) instead of CPU then add the `--gpus all` option:

```
docker run --name merlin-training [--gpus all] -d -it --net=host -p 8888:8888 -p 8797:8787 -p 8796:8786 -p 8889:8889 -v $(pwd)/Merlin/examples/Building-and-deploying-multi-stage-RecSys:/Merlin/examples/Building-and-deploying-multi-stage-RecSys --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-tensorflow:23.12 /bin/bash -c "cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token=''"
```

View the jupyter notebook server at http://localhost:8888

### Serving Container

If you want to use GPU(s) instead of CPU then add the `--gpus all` option:

```
docker run --name merlin-serving [--gpus all] --restart=on-failure -d -it --net=host -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $(pwd)/Merlin/examples/Building-and-deploying-multi-stage-RecSys:/Merlin/examples/Building-and-deploying-multi-stage-RecSys --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-tensorflow:23.12 /bin/bash -c "pip install feast==0.31; tritonserver --model-repository=/Merlin/examples/Building-and-deploying-multi-stage-RecSys/poc_ensemble --backend-config=tensorflow,allow-soft-placement=true --model-control-mode=poll --repository-poll-secs=5"
```

## Documentation

https://github.com/NVIDIA-Merlin/models/tree/main/examples

https://github.com/NVIDIA-Merlin/Merlin/tree/main/examples
