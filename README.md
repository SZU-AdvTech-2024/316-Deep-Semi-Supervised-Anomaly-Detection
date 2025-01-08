# Deep Semi-Supervised Anomaly Detection (ICLR 2020)

This is the reproduced version of the [original PyTorch implementation](https://github.com/lukasruff/Deep-SAD-PyTorch) of the ICLR paper.

## Citation
If you find this code to be useful for your research, please consider citing.

```
@InProceedings{ruff2020deep,
  title     = {Deep Semi-Supervised Anomaly Detection},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Binder, Alexander and M{\"u}ller, Emmanuel and M{\"u}ller, Klaus-Robert and Kloft, Marius},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  url       = {https://openreview.net/forum?id=HkgH0TEYwH}
}
```

## Setup 
This code is written in `Python 3.10`,`CUDA = 12.4` and requires the packages listed in `requirements.txt`.


To run the code, we recommend setting up a virtual environment, e.g. using `conda`:

### `conda`
```
cd <path-to-Deep-SAD-PyTorch-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```
## Dataset
This experiment uses three datasets: MNIST, Fashion-MNIST, and CIFAR-10. 
If you want to verify more other datasets, you can refer to other common anomaly detection datasets, 
such as: `arrhythmia`, `cardio`, `satellite`, `satimage-2`, `shuttle`, and `thyroid`.

### Deep SAD
You can run Deep SAD experiments using the `main.py` script.    

Here's an example on `MNIST` with `0` considered to be the normal class and having 1% labeled (known) training samples 
from anomaly class `1` with a pollution ratio of 10% of the unlabeled training data (with unknown anomalies from all 
anomaly classes `1`-`9`):
```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folders for experimental output
mkdir log/DeepSAD
mkdir log/DeepSAD/mnist_test

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/DeepSAD/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --seed 42 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
```
Have a look into `main.py` for all possible arguments and options.

### Baselines
We also provide an implementation of the following baselines via the respective `baseline_<method_name>.py` scripts:
OC-SVM (`ocsvm`), Isolation Forest (`isoforest`), Kernel Density Estimation (`kde`), kernel Semi-Supervised Anomaly 
Detection (`ssad`), and Semi-Supervised Deep Generative Model (`SemiDGM`).

Here's how to run SSAD for example on the same experimental setup as above:
```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/ssad
mkdir log/ssad/mnist_test

# change to source directory
cd src

# run experiment
python baseline_ssad.py mnist ../log/ssad/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --seed 42 --kernel rbf --kappa 1.0 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
```

The autoencoder is provided through Deep SAD pre-training using `--pretrain True` with `main.py`. 
To then run a hybrid approach using one of the classic methods on top of autoencoder features, simply point to the saved
autoencoder model using `--load_ae ../log/DeepSAD/mnist_test/model.tar` and set `--hybrid True`.

To run hybrid SSAD for example on the same experimental setup as above:
```
cd <path-to-Deep-SAD-PyTorch-directory>

# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# create folder for experimental output
mkdir log/hybrid_ssad
mkdir log/hybrid_ssad/mnist_test

# change to source directory
cd src

# run experiment
python baseline_ssad.py mnist ../log/hybrid_ssad/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --seed 42 --kernel rbf --kappa 1.0 --hybrid True --load_ae ../log/DeepSAD/mnist_test/model.tar --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
```
## Acknowledgements
Thanks to the author of [Deep Semi-Supervised Anomaly Detection](https://arxiv.org/abs/1906.02694) for providing such anomaly detection ideas and making the code public for us to learn.

