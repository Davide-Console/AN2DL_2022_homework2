AN2DL_2022_homework2
Berzoini Raffaele, Console Davide, Manzo Noemi

Our project has been developed with TensorFlow 2.4.1 and python 3.9.15pr

In a terminal execute:
```bash
git clone https://github.com/noemimanzo/AN2DL_2022_homework1.git
cd AN2DL_2022_homework1
```
## Networks
To visualize the architecture of pur best model execute
```bash
python networks.py
```
## Prepare dataset
Before training, prepare the dataset executing:
```bash
python dataset_preparation.py
```
## Training
To perform training with time-series of 6 features (default):
```bash
python variable_training.py
```

To perform training with time-series of 5 features:
```bash
python variable_training.py -mod 5_features
```

To perform training with data augmentation:
```bash
python variable_training.py -mod data_aug
```

To perform training with 2D reshaped time-series:
```bash
python variable_training.py -mod 2d
```

To perform training with fft added features:
```bash
python variable_training.py -mod fft
```
#


---
GPU_MEMORY and execution_settings are utils scripts to perform training on a local GPU (RTX2060 Mobile and a GTX1050 Mobile)