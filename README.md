# poincare_flow_matching
# Installation

```
conda create -n hypfm  python=3.10
conda activate hypfm
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge pytorch-lightning torchdiffeq  matplotlib h5py timm diffusers accelerate loguru \
blobfile hydra-core wandb einops scikit-learn transformers pycocotools absl-py
pip install ml_collections
```

# Installation for toy_examples
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning-bolts matplotlib numpy scipy scikit-learn scanpy timm pot -c conda-forge
pip install torchdyn>=1.0.5 torchdiffeq==0.2.3 scprep
```
