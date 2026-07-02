# ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts

### [Project Page](https://lodurality.github.io/shapewords/) | [Paper (arXiv)](https://arxiv.org/abs/2412.02912)

**Official PyTorch implementation of ShapeWords (CVPR 2025)**

ShapeWords incorporates target 3D shape information into text prompts for guided image synthesis. Given a 3D shape (encoded with PointBERT tokens) and a text prompt, our Shape2CLIP module predicts a shape-aware offset to the CLIP prompt embedding that guides Stable Diffusion towards images that comply both with the target 3D shape and the text prompt.

## Setup

Create the conda environment:

```
conda env create -f environment.yml
conda activate shapewords_source
```

## Quickstart

### Pretrained checkpoint

Download the pretrained Shape2CLIP checkpoint [here](https://drive.google.com/uc?id=1nvEXnwMpNkRts6rxVqMZt8i9FZ40KjP7), or via `gdown`:

```
gdown 1nvEXnwMpNkRts6rxVqMZt8i9FZ40KjP7 -O projection_model-0920192.pth
```

The checkpoint corresponds to `Shape2CLIP(depth=6, heads=8, pb_dim=384)` trained on PointBERT shape tokens (see `geometry_guidance_models.py`).

### Demo

We provide a Gradio demo in the `demo/` folder (also deployable as a Hugging Face ZeroGPU space). It downloads the Shape2CLIP checkpoint automatically on first run:

```
pip install -r demo/requirements.txt
cd demo && python app.py
```

The demo expects per-category PointBERT embedding files (`<synset_id>_pb_embs.npz`) in `demo/embeddings/` — see the sample file in `sample_data/shapenet_pointbert_tokens/` for the expected format (`ids` and `embs` arrays).

## Training

Since our dataset is fairly large, we provide a command to run training on sample data (a few shapes of the `02773838` (bag) category are included in `sample_data/`). For full-scale training, download our data and replace the paths accordingly — see [Data](#data) below.

To train the Shape2CLIP guidance model on sample data run the following:

```
bash ./train_on_sample_data.sh
```

Model checkpoints will be saved in `sample_outputs/`.

To run full-scale training on a SLURM cluster, fill in `data_root`, the SBATCH header fields and `HF_HOME` in `train_on_cluster.sh` and submit it:

```
sbatch ./train_on_cluster.sh
```

Training is roughly based on the Hugging Face diffusers [textual inversion example](https://huggingface.co/docs/diffusers/en/training/text_inversion): we freeze the Stable Diffusion 2.1 VAE, U-Net and text encoder, and optimize only the Shape2CLIP model with the denoising objective, with optional timestep-dependent loss weighting (`--weight_loss_by_t`).

## Data

The full preprocessed training data is available here:

https://console.cloud.google.com/storage/browser/shapewords_data

The training data layout (see `sample_data/` for a working example):

`controlnet_images_offset_all/<synset_id>/<synset_id>_<shape_id>/combined/` — ControlNet-generated training images per shape (`angle_<view>_prompt_<prompt_id>_combo_<k>.jpg`)

`controlnet_images_offset_all/<synset_id>/<synset_id>_<shape_id>/depth/` — depth renders per view (`depth_<view>.jpg`)

`shapenet_pointbert_tokens/<synset_id>_pb_embs.npz` — PointBERT shape tokens per category (`ids` and `embs` arrays)

`foreground.txt` — stylized text prompt templates used during training

`categories.json` — mapping from ShapeNet synset ids to category names

Train/val/test splits are provided in `stats/` (`train.txt`, `val.txt`, `train_val.txt`, `test.txt`), one `<synset_id>_<shape_id>` per line.

If you have any questions about the data, feel free to open an issue or email me (Dmitrii Petrov).

## Citation

If you find our work useful, please cite the CVPR 2025 paper:

```bibtex
@InProceedings{Petrov_2025_CVPR,
    author    = {Petrov, Dmitry and Goyal, Pradyumn and Shivashok, Divyansh and Tao, Yuanming and Averkiou, Melinos and Kalogerakis, Evangelos},
    title     = {ShapeWords: Guiding Text-to-Image Synthesis with 3D Shape-Aware Prompts},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {13305-13314}
}
```

## Acknowledgements

The attention implementation in `geometry_guidance_models.py` is based on [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet) by [Biao Zhang](https://1zb.github.io/) and co-authors. The training script is roughly based on the Hugging Face diffusers [textual inversion example](https://huggingface.co/docs/diffusers/en/training/text_inversion). Shape embeddings are produced with PointBERT encoders from [PointBERT](https://github.com/Julie-tang00/Point-BERT).
