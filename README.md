# SPREAD
Code for SPREAD: Spatial-Physical REasoning via geometry Aware Diffusion

## Environment Setup

Before running the code, please make sure:

1. **CUDA is properly installed** and matches your GPU driver.
2. Create a virtual environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate my_env
```

Next, please navigate to the `torch-mesh-isect` directory and follow the installation instructions provided by that module:

```bash
cd torch-mesh-isect
python setup.py install
```

After installing all dependencies, you may want to download and use our trained checkpoint from this [Anonymous Links](https://doi.org/10.5281/zenodo.16652807), and put the `ckpt.pth` in the path of `./out/example/checkpoints`.

Then return to the root directory and run the inference script using the following format:

```bash
cd ..
bash scripts/inference_sg2sc_image_new.sh CONFIG_NAME CUDA_DEVICE_ID
```

For instance, you can run it with the provided example with correct checkpoint:
```bash
bash scripts/inference_sg2sc_image_new.sh procthor_graph_point_same_lat_parallel_fixed_cos example 0
```