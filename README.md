# GDAL: Gradient Disagreement in Multiple Quantile Regression for Active Learning
The source code of the GDAL method.
An AL method that scores unlabeled 3D molecules by the disagreement among the per-quantile gradients of a multi-quantile regression head.

## Dependencies
Please find the dependencies in the `environment.yml` file (conda environment name: `3dgraph`).

```bash
conda env create -f environment.yml
conda activate 3dgraph
```

## Code Structure
The code is structured as follows:
- `al/selection_methods.py`: The active learning methods. `gradient_conflicts` is the main implementation of our method.
- `al/quantile_training.py`: The multi-quantile regression (MQR) head and its training loop.
- `al/sampler.py`: The sequential subset sampler used to keep the order of the unlabeled pool.
- `models`: The 3D backbones. `spherenet_bgnn.py` and `dimenetpp_bgnn.py` are the Bayesian variants used for training and for extracting molecular embeddings.
- `datasets`: The dataset interface and the labeled/unlabeled split logic.
- `utils`: The training/evaluation loop (`run.py`), the metrics (`eval.py`) and the molecular similarity matrices (`compute_similarity_matrix_*.py`).
- `dataset/qm9/processed`: The folder holding the processed `.pt` dataset files.
- `runs/{selection_method}/run{expt}`: The folder to save the experimental results, the checkpoints and the current labeled set (`init_set.npy`).

## Usage
Put the processed dataset in `dataset/qm9/processed/` and the initial labeled set in `runs/{selection_method}/run{expt}/init_set.npy`, then run one full AL experiment with:

```bash
bash run.sh
```

The target property, backbone, query batch size (`ADDENDUM`), pool size and number of AL cycles are configured at the top of `run.sh`. A single cycle can also be launched directly:

```bash
python3 train.py --selection_method gradient_conflicts --backbone spherenet --target homo --cycle 0 --expt 1
```

The `unc_div` baseline additionally requires a precomputed similarity matrix, which is produced by `compute_sim_mat.py`.

## License
The code implementation is licensed under the Apache 2.0 license.

## Comparison Methods Code
The overall 3D graph active learning pipeline and baselines are adapted from https://github.com/sronast/al_3dgraph.

The SphereNet and DimeNet++ backbones are adapted from the DIG library, https://github.com/divelab/DIG.

The MQR-UD active learning method is adapted from https://github.com/DongNan-ML/mqrud

## Datasets
The QM9 dataset comes from the https://pytorch-geometric.readthedocs.io/en/latest/

The Qmugs dataset comes from the https://www.openqdc.io/


## Citations: the comparision methods and some utility functions in this repository are from:

```bibtex

@article{subedi2024empowering,
  title={Empowering active learning for 3D molecular graphs with geometric graph isomorphism},
  author={Subedi, Ronast and Wei, Lu and Gao, Wenhan and Chakraborty, Shayok and Liu, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={55507--55537},
  year={2024}
}

@article{sener2017active,
  title={Active learning for convolutional neural networks: A core-set approach},
  author={Sener, Ozan and Savarese, Silvio},
  journal={In International Conference on Learning Representations},
  year={2018}
}

@inproceedings{liu2022spherical,
  title={Spherical message passing for 3d molecular graphs},
  author={Liu, Yi and Wang, Limei and Liu, Meng and Lin, Yuchao and Zhang, Xuan and Oztekin, Bora and Ji, Shuiwang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2022}
}

@article{gasteiger2020fast,
  title={Fast and uncertainty-aware directional message passing for non-equilibrium molecules},
  author={Gasteiger, Johannes and Giri, Shankari and Margraf, Johannes T and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:2011.14115},
  year={2020}
}

@article{JMLR:v22:21-0343,
  author  = {Meng Liu and Youzhi Luo and Limei Wang and Yaochen Xie and Hao Yuan and Shurui Gui and Haiyang Yu and Zhao Xu and Jingtun Zhang and Yi Liu and Keqiang Yan and Haoran Liu and Cong Fu and Bora M Oztekin and Xuan Zhang and Shuiwang Ji},
  title   = {{DIG}: A Turnkey Library for Diving into Graph Deep Learning Research},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {240},
  pages   = {1-9},
  url     = {http://jmlr.org/papers/v22/21-0343.html}
}

@article{ramakrishnan2014quantum,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and Von Lilienfeld, O Anatole},
  journal={Scientific data},
  volume={1},
  number={1},
  pages={1--7},
  year={2014},
  publisher={Nature Publishing Group}
}

@article{isert2022qmugs,
  title={QMugs, quantum mechanical properties of drug-like molecules},
  author={Isert, Clemens and Atz, Kenneth and Jim{\'e}nez-Luna, Jos{\'e} and Schneider, Gisbert},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={273},
  year={2022},
  publisher={Nature Publishing Group UK London}
}

@article{gabellini2024openqdc,
  title={OpenQDC: Open Quantum Data Commons},
  author={Gabellini, Cristian and Shenoy, Nikhil and Thaler, Stephan and Canturk, Semih and McNeela, Daniel and Beaini, Dominique and Bronstein, Michael and Tossou, Prudencio},
  journal={arXiv preprint arXiv:2411.19629},
  year={2024}
}


```
