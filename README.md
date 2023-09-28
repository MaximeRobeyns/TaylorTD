# TaylorTD

Providing lower variance TD updates through a first-order Taylor expansion of expected TD updates.

# Installation Instructions

## Installing MuJoCo

The project uses OpenAI Gym environments (e.g. `HalfCheetah-v2`), which depend
on MuCoJo `1.50` specifically (through `mujoco-py`). In addition, it apparently
_must_ be installed under `~/.mujoco/mjpro150` (despite the
`MUJOCO_PY_MUJOCO_PATH` env var apparently indicating otherwise). Nonetheless,
to install it (on linux), run:

``` sh
mkdir -p ~/.mujoco && cd ~/.mujoco
curl -LO https://roboti.us/download/mjpro150_linux.zip
unzip mjpro150_linux.zip && rn mjpro150_linux.zip
```

## Shell Configuration and Virtual Environment

To keep things versioned and segregated from the rest of the system, we should
use a virtual environment. We will use a `conda` virtual environment called
`taylorrl` for this project.

``` sh
conda create [-p /optional/prefix] -n taylorrl
```

We will also need to set some environment variables. A convenient way to manage
per-project environment variables and other shell configurations is to use
[direnv](https://direnv.net/) (highly recommended!).

In the project's `.envrc` file, you can begin by activating the conda environment:

``` sh
conda activate taylorrl
```

Before installing the python dependencies (i.e. linking `mujoco-py` against the
dynamic library we just downloaded in the previous section), we need to set the
linker path so that `ld` knows where to find mujoco. In `.envrc`,


``` sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
```

I have also found that the linker cannot find the `GLIBCXX_3.4.29` symbol, which
can be rectified by preloading the OS's `libstdc++.so` file, by adding the
following to `.envrcc` (change the location to point to your `libstdc++.so`).

``` sh
export LD_PRELOAD=/usr/lib64/libstdc++.so.6
```

In order to render things, you will also need `GLEW`. Using anaconda

``` sh
conda install -c conda-forge glew mesalib patchelf gxx gcc
conda install -c anaconda mesa-libgl-cos6-x86_64 swig
conda install -c menpo glfw3
```

now, in `.envrc`:

``` sh
export LD_PRELOAD=</path/to/conda/env/>lib/libGLEW.so:$LD_PRELOAD
```

## Installing python packages

We will begin with PyTorch and CUDA libraries, since the CUDA libraries are
added to the linker path and may be used by other packages later on.

``` sh
conda install python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

We can now go ahead and install OpenAI Gym (and the corresponding environments):

``` sh
pip install gym[all]
```

And any other python packages required:

``` sh
pip install dotmap sacred
```
## Note 
The underlying structure of the code is based (but not forked) on [MAGE](https://github.com/nnaisense/MAGE): Model-based Action-Gradient-Estimator Policy Optimization ([paper](https://arxiv.org/abs/2004.14309)).
