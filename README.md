# *Coarse-to-Fine Spatio-Temporal Information Fusion for Compressed Video Quality Enhancement* (SPL 2022)
The PyTorch implementation for the CF-STIF: *[Coarse-to-Fine Spatio-Temporal Information Fusion for Compressed Video Quality Enhancement](https://drive.google.com/file/d/1RlFVFvP-VUMXgbbUHjLAmWarl4sEuCUR/view)* which is accepted by [IEEE SPL].
## 1. Pre-request
### 1.1. Environment
```bash
conda create -n CF-STIF python=3.10.11
conda activate CF-STIF
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

git clone --depth=1 https://github.com/xiaomingxige/CF-STIF
cd CF-STIF
pip install -r requirements.txt
```
### 1.2. DCNv2
#### Build DCNv2

```bash
cd ops/dcn/
# You may need to modify the paths of cuda before compiling.
bash build.sh
```
#### Check if DCNv2 works (optional)

```bash
python simple_check.py
```
> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility. [[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)

### 1.3. MFQEv2 dataset
#### 1.3.1 Download and compress videos
Please check [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset).
#### 1.3.2 Edit YML

We now edit `option_R3_mfqev2_1G.yml`.

Suppose the folder `MFQEv2_dataset/` is placed at `/raid/xql/datasets/MFQEv2_dataset/`, then you should assign `/raid/xql/datasets/MFQEv2_dataset/` to `dataset -> train -> root` in YAML.


> `1G`: 1 GPUs will be used for the below training. 
> Similarly, you can also edit `option_mfqev2_2G.yml` and `option_mfqev2_4G.yml` if needed.

#### 1.3.3 Generate LMDB

We now generate LMDB to speed up IO during training.

```bash
python create_lmdb_mfqev2.py --opt_path option_mfqev2_1G.yml
```


Finally, the MFQEv2 dataset root will be sym-linked to the folder `./data/` automatically.

> So that we and programmes can access MFQEv2 dataset at `./data/` directly.
## 2. Train
See `script.sh`.
## 3. Test

Pretrained models can be found here: [GoogleDisk](https://drive.google.com/drive/folders/152Spbw_uJ31ClImhPi9W91vNY9CGVTfg?usp=sharing) and [[百度网盘 (CF-STIF)]](https://pan.baidu.com/s/1I-c95lJYLNmIQALzqelWYA)
We utilize 1 NVIDIA GeForce RTX 3090 GPU for testing：

```bash
python test_video.py
```
## Citation
If you find this project is useful for your research, please cite:

```bash
@article{luo2022coarse,
  title={Coarse-to-fine spatio-temporal information fusion for compressed video quality enhancement},
  author={Luo, Dengyan and Ye, Mao and Li, Shuai and Li, Xue},
  journal={IEEE Signal Processing Letters},
  volume={29},
  pages={543--547},
  year={2022},
  publisher={IEEE}
}
```
## Acknowledgements
This work is based on [STDF-Pytoch](https://github.com/ryanxingql/stdf-pytorch). Thank [RyanXingQL](https://github.com/RyanXingQL) for sharing the codes.
