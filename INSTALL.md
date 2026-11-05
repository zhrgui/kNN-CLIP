## Installation

The results in the paper were obtained on Linux with an NVIDIA A40, Python 3.9, PyTorch 2.0.0 and CUDA 11.8. The steps below set up that environment.

### Example conda environment setup

```bash
conda create --name knnclip -c conda-forge python=3.9 "numpy<2" faiss-cpu=1.9.0 -y
conda activate knnclip
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118

# under your working directory
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install --no-build-isolation --use-pep517 -e detectron2
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
git clone https://github.com/zhrgui/kNN-CLIP.git
cd kNN-CLIP
pip install -r requirements.txt
```

A few of these choices matter:

- FAISS comes from conda-forge because its build gets faster with more threads, which the PyPI wheels do not. On 4 cores, searching the A-150 database for one image takes about 0.4 s, against 2 to 7 s with the PyPI wheels. See [FAISS threads](#faiss-threads) for the setting this relies on.
- `--no-build-isolation --use-pep517` lets the detectron2 build see the PyTorch that is already installed.
- `requirements.txt` pins `open_clip_torch`, as later versions change the text encoder interface that FC-CLIP relies on.

### CUDA toolkit

The next step needs a CUDA toolkit whose version matches your PyTorch, which is 11.8 for the commands above. Conda can install one into the environment:

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Keep `LD_LIBRARY_PATH` set whenever you use the environment. PyTorch's cuDNN loads `libnvrtc.so` from this toolkit at run time, and without it fails with `Could not load library libcudnn_cnn_infer.so.8`.

### CUDA kernel for MSDeformAttn

Compile the kernel on a machine with a GPU:

```bash
cd fcclip/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
```

On a machine without a GPU device, replace `sh make.sh` with:

```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

### FAISS threads

Before evaluating, set `OMP_NUM_THREADS` to the number of physical CPU cores available to each process. Hyper-threads do not count, so an allocation of 8 logical CPUs on a machine with 2 threads per core means 4:

```bash
export OMP_NUM_THREADS=4
```

Search slows down sharply beyond the physical core count, and leaving the variable unset uses every logical CPU. On 4 cores with 8 hardware threads, searching the A-150 database for one image takes about 0.4 s with 4 threads and 7 to 12 s when unset.
