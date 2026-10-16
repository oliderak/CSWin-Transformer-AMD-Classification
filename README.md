# CSWin Transformer for Age-Related Macular Degeneration Classification

Research code and experimental methodology from a Master's thesis in Artificial Intelligence at the Universidad Autónoma de Querétaro (May 2026): **“Clasificación automática de degeneración macular asociada a la edad en imágenes de tomografías de coherencia óptica mediante aprendizaje profundo.”**

This project investigates **hybrid CSWin Transformer architectures** for classifying retinal optical coherence tomography (OCT) B-scans into three categories: **NORMAL**, **DRUSEN**, and **CNV** (choroidal neovascularization). It studies two multiscale feature aggregation strategies: **Feature Pyramid Network (FPN)** and **Atrous Spatial Pyramid Pooling (ASPP)**.

> **Repository scope:** Both **CSWin + FPN** and **CSWin + ASPP** are included as separate training/evaluation notebooks. Each defines the hybrid model, loads patient-aware fold indices from the common preprocessed dataset, trains across five folds, saves per-fold checkpoints, and evaluates the model. The results below are those reported in the thesis; reproducing them requires the dataset, dependencies, and experiment configuration.

## Research objectives

- Investigate CSWin Transformer as a hierarchical feature extractor for grayscale retinal OCT images.
- Compare FPN and ASPP as multiscale modules for distinguishing NORMAL, DRUSEN, and CNV patterns.
- Build a consistent OCT preprocessing workflow and use patient-aware data partitioning.
- Evaluate classification performance with accuracy, precision, recall, F1 scores, and one-vs-rest ROC-AUC.
- Explore model behavior through Grad-CAM and deep feature activation visualizations.

## Model architectures

**CSWin + FPN:** A CSWin Transformer backbone extracts hierarchical visual features. An FPN combines features across spatial resolutions through lateral connections and top-down fusion, followed by feature aggregation and a classification head.

**CSWin + ASPP:** A CSWin Transformer backbone is coupled with parallel atrous convolutions and global context aggregation before classification. The thesis identified this configuration as the strongest of the evaluated variants.

The underlying CSWin Transformer backbone was proposed by Dong et al.; this research investigates its adaptation and integration into OCT classification pipelines rather than claiming authorship of the original CSWin Transformer.

## Dataset

The research uses the public **Labeled Retinal Optical Coherence Tomography Dataset for Classification of Normal, Drusen, and CNV Cases** by Sotoudeh-Paima and collaborators, available through Mendeley Data.

- Original collection: **16,822 OCT B-scans from 441 patients**.
- After the thesis's metadata-based within-patient filtering: **12,649 images**, retaining 441 patients.
- Classification labels: `NORMAL`, `DRUSEN`, `CNV`.

**Data availability:** Download the dataset from its original source and comply with its attribution requirements. Images are not included in this repository unless explicitly stated otherwise.

## Dataset splitting and preprocessing workflow

The dataset preparation has **two separate, sequential notebooks**. Both are essential to understand the experiment and should be run before either model notebook.

### Step 1 — `1DivisionDataset.ipynb`: inspect, filter and split

- Check the original dataset contents, image dimensions, and CSV metadata (`data_information.csv`).
- Match individual B-scan labels to the corresponding patient-level classification to perform the thesis's deterministic within-patient filtering (16,822 original images reduced to 12,649 images from 441 patients).
- Construct patient-level partitions to avoid overlap of patient identities between training and external testing.
- Reserve **20% of the data as an external patient-level test partition** according to the research protocol.
- Build the **stratified, patient-group-aware 5-fold cross-validation splits** on the remaining training partition.

### Step 2 — `2DatasetUnico.ipynb`: create one reusable preprocessed dataset

- Load and validate the partition and fold metadata generated in Step 1, including structural sanity checks.
- Apply the OCT preprocessing pipeline: grayscale reading, ROI-guided cropping, geometric standardization to **512 × 512** with aspect ratio preservation and padding, and valid-region z-score normalization.
- Materialize a **single reusable base dataset** as stored shards, rather than rebuilding image data separately for every fold.
- Verify the materialized data and generate fold-specific indices referencing the common base dataset.
- Confirm that external test data and fold assignments remain consistent with the patient-aware experimental protocol.

**Why these steps matter:** the model notebooks read the materialized base dataset and its indices; they do not replace the initial data preparation. Data augmentation is performed in the training workflow rather than being permanently applied to validation or test data.

## Repository notebooks

| Notebook | Role | Notes |
| --- | --- | --- |
| `1DivisionDataset.ipynb` | Original image/metadata checks, within-patient filtering, patient-level external test split and grouped fold definition. | Run first. |
| `2DatasetUnico.ipynb` | ROI-aware preprocessing, single materialized dataset in shards, consistency checks and per-fold indices. | Requires the outputs of Step 1. |
| `3CSWinTransformerFPN_A100.ipynb` | CSWin + FPN architecture, five-fold training and validation, evaluation, per-fold checkpoints and visualizations. | Uses the common base dataset. |
| `3CSWinTransformerASPP_A100.ipynb` | CSWin + ASPP architecture, five-fold training and validation, evaluation, per-fold checkpoints and visualizations. | Uses the common base dataset; includes the ASPP training pipeline. |

## Getting started

1. Use **Google Colab** or a compatible Python/PyTorch environment with GPU access. The thesis experiments used an NVIDIA A100.
2. Download the original OCT dataset and its `data_information.csv` metadata from the cited Mendeley Data source; these files are not automatically included.
3. Run `1DivisionDataset.ipynb` to inspect/filter the data and create the patient-aware split and fold files.
4. Run `2DatasetUnico.ipynb` to build and validate the reusable preprocessed dataset and its stored fold indices.
5. For the **FPN experiment**, open `3CSWinTransformerFPN_A100.ipynb`; configure data paths, pretrained CSWin dependencies, output directories and GPU resources, then run its five-fold training, evaluation and visualization cells.
6. For the **ASPP experiment**, open `3CSWinTransformerASPP_A100.ipynb`; use the **same shared dataset and fold assignments**, configure its corresponding paths and GPU resources, then run its five-fold training, evaluation and visualization cells. The two experiments can be run separately after Steps 1–2.

The notebooks depend on the Google Drive / Colab directory layout used for the thesis (including paths under `/content/drive/MyDrive/TesisIA` and the `OCT_base_cv5` data directory). Update these paths for your environment. Python dependencies include PyTorch and notebook-specific packages such as `timm`, `einops`, `opencv-python`, `scikit-image`, `scikit-learn` and `matplotlib`. Consult the setup cells for the exact requirements.

The underlying CSWin Transformer implementation and pretrained weights may need to be obtained separately, subject to their respective licenses. **These notebooks are research code and may require path/configuration adjustments; cloning the repository alone does not guarantee one-click reproduction.**

## Results reported in the thesis

The best-performing evaluated configuration was **CSWin Transformer + ASPP**, with the following reported mean metrics (± standard deviation):

| Metric | CSWin + ASPP (base training) |
| --- | ---: |
| Accuracy | **0.9440 ± 0.0073** |
| Macro-F1 | **0.9433 ± 0.0069** |
| Macro-AUC | **0.9799 ± 0.0014** |

These figures are **research results reported in the thesis**; the ASPP training notebook contains the implementation needed to attempt reproduction, but exact results depend on the dataset, environment, and configuration. Interpret comparisons in the context of the specific dataset and evaluation protocol; they do not establish clinical deployment readiness.

## Limitations and intended use

This repository is intended for **academic research and reproducibility**, not clinical diagnosis or patient-care decisions. Its results were obtained on the research dataset and protocol described above; clinical application would require further independent validation.

## References and attribution

- **Master's thesis:** Castillo Rocha, Oliverio (2026). *Clasificación automática de degeneración macular asociada a la edad en imágenes de tomografías de coherencia óptica mediante aprendizaje profundo*. Universidad Autónoma de Querétaro.
- **CSWin Transformer:** Dong et al. (2022). *CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows*. [Official implementation](https://github.com/microsoft/CSWin-Transformer).
- **Dataset:** Sotoudeh-Paima et al. *Labeled Retinal Optical Coherence Tomography Dataset for Classification of Normal, Drusen, and CNV Cases*. Mendeley Data. Consult the dataset record for its exact citation and license terms.
- **FPN:** Lin et al. (2017). *Feature Pyramid Networks for Object Detection*.
- **ASPP:** Chen et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*.

