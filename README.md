# PhenoLearn

‘PhenoLearn’ (github.com/EchanHe/PhenoLearn) is an open-source image analysis tool that is designed for high-throughput phenotype measuring on digital biodiversity datasets. It can (i) generates annotations  (currently points and segmentation) (ii) use deep learning models to train and predict annotations (iii) Review and edit predictions.

  

## Prerequisites

- Python >= 3.6

- QtPy >= 5.97

- numpy >= 1.20.3

- pandas >= 1.3.4

- opencv-python = 4.5.5.64

- tensorflow = 1.6.0
  
  

## Installation

  Clone the repo

```bash
git clone https://github.com/EchanHe/PhenoLearn.git
```



## Usage

### Annotation and the review
To use the annotation and the review function. Please run mainWin.py.
 ```bash
python mainWin.py
```

Or you can create a binary version using
  
 ```bash
pyinstaller mainWin.py
```

### Deep Learning
TODO

## TODO
- Update the backbone using newer version of deep learning libraries.
- Enable read in image-based mask annotations.