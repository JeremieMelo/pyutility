# pyutils
A python/pytorch utility library

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/JeremieMelo/pyutility/blob/release/LICENSE)

## News
- v0.0.1 available. Feedbacks are highly welcomed!

## Installation
```bash
git clone https://github.com/JeremieMelo/pyutility.git
cd pyutility
pip3 install --editable .
```

## Features
- Support pytorch training utility and datasets.

## TODOs
- [ ] Support lr_scheduler
- [ ] Support trainer

## Dependencies
- Python >= 3.6
- PyTorch >= 1.8.0
- Tensorflow >= 2.5.0
- Others are listed in requirements.txt


## Files
| File      | Description |
| ----------- | ----------- |
| datasets/ | Defines different datasets and builder |
| optimizer/ | Defines different optimizers |
| activation.py      | Activation functions |
| compute.py   | functions related to computing |
| config.py   | Hierarchical yaml configuration file parser |
| distribution_sampler.py   | Sample from customized distributions |
| general.py   | Common helper functions |
| initializer.py   | Initialization methods for PyTorch Parameters |
| loss.py   | Loss functions for PyTorch model training |
| quantize.py   | Quantization functions |
| torch_train.py   | Helper functions for torch training |
| typing.py | Defines common types |


## Contact
Jiaqi Gu (jqgu@utexas.edu)
