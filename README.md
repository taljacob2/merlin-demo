# merlin-demo

A Demo for [NVIDIA Merlin](https://developer.nvidia.com/merlin) - A framework for training & serving Deep Learning Recommendation Models.

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

```
for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done
```

> This mutes the non-fatal warnings of:
> ```
> 2025-03-22 12:25:38.494146: I external/local_xla/xla/stream_executor/cuda/cuda_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero. See more at https://github.com/torvalds/linux/blob/v6.0/Documentation/ABI/testing/sysfs-bus-pci#L344-L355
> ```
> See https://github.com/tensorflow/tensorflow/issues/42738#issuecomment-922422874

```
docker run --gpus all --rm -it -p 8888:8888 -p 8797:8787 -p 8796:8786 --ipc=host --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-tensorflow:23.12 /bin/bash -c "cd / ; jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token=''"
```

View the jupyter notebook server at http://localhost:8888

## Documentation

https://github.com/NVIDIA-Merlin/models/tree/main/examples

