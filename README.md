# PhenoLearn

**PhenoLearn** is an open-source image analysis tool that is designed for high-throughput phenotype measuring on digital biodiversity datasets. It can (i) generates annotations (ii) use deep learning models to generate annotations (iii) Review predictions.

## Table of Contents
-   [Features](#features)
-   [Installation](#installation)
-   [Get demo datasets](#get-demo-datasets)
-   [Usage](#usage)
-   [License](#license)
-   [Contact](#contact)  

## Features

PhenoLearn has two UI tools:
1. **PhenoLabel**. does the Label and Review
2. **PhenoTrain**. does the Deep Learning

Functions:
- **Labeling**: Place labels on images, currently supports points and segmentations.
- **Using Deep Learning**: Train DL models with labelled images, and apply the best model to make predictions.
- **Review**: Review the predictions, edit incorrect predictions to increase the accuracy


PhenoLearn can be used in different workflows

![Main view](./assets/workflow.png)

## Installation
Before you start the installation, make sure your system meets the following requirements:

-  Python 3 (Tested Versions: 3.11) installed. If you don't have Python installed, you can get it from [Python](https://www.python.org/downloads/) or [Anaconda Python](https://www.anaconda.com/download). We recommend Anaconda Python, as it has Conda included.
-   pip (Python Package Installer), which typically comes with the Python installation
-   Conda, an open-source package management system and environment management system
  
To install, follow these steps.
<br />

**1. Clone the repo**
 
```bash
git clone https://github.com/EchanHe/PhenoLearn.git
```
It is recommended to create a virtual environment to avoid mixing up the project dependencies with other projects. You can create a virtual environment using the following command:
<br />

**2. Create a new Conda environment and activate it:**

The python version we developed PhenoLearn is 3.10.

```bash
conda create --name phenolearn python=3.10 
conda activate phenolearn
```
You can also custom your environment. For further information about virtual environment you can visit [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)
<br />

**3. Install pacakges:**

When you are in the `phenolearn` virtual environment, install packages for PhenoLearn.

Required packages for PhenoLabel with tested versions:
- numpy == 1.25.1
- pandas == 2.0.3
- opencv-python == 4.8.0.74
- PyQt == 5.15.9
- TensorBoard == 2.13.0

Other versions may also work, but they have not been tested.\
Use the `requirements.txt` in the root directory to install
 ```bash
pip install -r requirements.txt
```
You can also install these packages directly using pip:
```bash
pip install numpy==1.25.1 pandas==2.0.3 opencv-python==4.8.0.74 PyQt5==5.15.9 TensorBoard == 2.13.0
```
<br />

We used PyTorch, a deep learning library, to implement PhenoLearn. 

The initial version was 2.0.1, and we have also tested our work on 2.2.2, which was the latest stable version as of 2024/04/02.

To install PyTorch, it depends on whether your device has a CUDA-supported GPU. You must install the correct version of `PyTorch` accordingly. For detailed installation instructions, please visit [the official website](https://pytorch.org/get-started/locally/). For installing previous versions, visit [here](https://pytorch.org/get-started/previous-versions/).
<br />

**Note**: This error could occur when trying to run PhenoLearn
```
QObject::moveToThread: Current thread (<Thread_id>) is not the object's thread (<Thread_id>).

Cannot move to target thread (<Thread_id>)

qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in <path_to_your_environment>/site-packages/cv2/qt/plugins even though it was found.

This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```
This error often happens in Linux. Visit [here](https://github.com/NVlabs/instant-ngp/discussions/300) for detailed information about this error.

To solve this, uninstall `opencv-python` and install `opencv-python-headless`
```
pip uninstall opencv-python
pip install opencv-python-headless
```


<br />






**4. Run PhenoLearn**

#### PhenoLabel
Navigate to the repository in the terminal and run 
 ```bash
python phenolabel.py
```

Or open the repo using [Visual studio Code](https://code.visualstudio.com/)
The run the `phenolabel.py` file

<br />

Or for Windows users, a binary version of PhenoLabel (e.g., an .exe file) is available at https://zenodo.org/records/10909841.

<br />

PhenoLearn provides an intuitive interface to easily label images for Deep Learning purposes.\
The window when PhenoLabel starts![Main view](./assets/main.png)


<br />

#### PhenoTrain
Navigate to the repository in the terminal and run 
 ```bash
python phenotrain.py
```

Or open the repo using [Visual studio Code](https://code.visualstudio.com/)
The run the `phenotrain.py` file
<br />
PhenoTrain provides an intuitive interface for using Deep Learning.\
The window when PhenoTrain starts![Main view](./assets/train.png)


## Get demo datasets

**1. 10 Bird images**

This dataset is designed for testing PhenoLearn:

Location: `./data/demo/`
- The `bird_10` folder has 10 images. They can be used for labeling and training purposes.
- `bird_10_process.json` is a saved labeling process file that can be loaded into PhenoLabel.
- `pts.csv` is a point CSV file. It stores image names in `bird_10/` with three points on them. It can be used in PhenoTrain training
- `seg.csv` is a segmentation CSV file. It stores image names in `bird_10/` with one segmentation. It can be used in PhenoTrain training
- The `valid_10` folder has 10 images. They can be used for PhenoTrain prediction.
- `valid_10_name.csv` provides the filenames for each image in `valid_10/`, useful for PhenoTrain predictions.
- `valid_10_props.csv` is a demo property file with properties for each image in `valid_10/`, which can be imported into PhenoLabel.

**2. The 220 bird images used in the paper**

120 images are in the `train` folder\
100 images are in the `pred` folder\
The segmentation CSV is `seg_train.csv`
The name file for prediction images is `name_file_pred.csv`

**3. The 220 *Littorina* images**

120 images are in the `train` folder\
100 images are in the `pred` folder\
The Point CSV file is `pts_train.csv`
The name file for prediction images is `name_file_pred.csv`

Both dataset 2 and 3 can be download here: https://zenodo.org/records/10988826

## Usage


-   [Label Image](#label-image)
-   [Use Deep Learning](#use-deep-learning)
-   [Review Predictions](#review-predictions)

### **Label Image**
This part uses [PhenoLabel](#phenolabel)

-   [Open a project](#open-a-project)
-   [Placing points](#placing-points)
-   [Placing segmentations](#placing-segmentations)
-   [Fast Labelling](#fast-labelling)  
-   [Saving and Loading Labeling Progress](#saving-and-loading-labeling-progress) 
-   [Export labels](#export-labels) 

#### **Open a project**

Click on the `Open Dir` in the `File menu` to select a directory containing the images to be labeled.
![open 2](./assets/open_1_1.png)

After selecting the directory, PhenoLearn will display all the images in the selected folder in the File panel
![open 2](./assets/opened_1.png)

Click on an image to display it in the Main panel. Here, you can inspect the image and zoom in or out using functions in `View menu`, or use Ctrl mouse wheels. The coordinates and RGB value of the current mouse location are displayed in bottom right
<br />

#### **Placing points**

Activate the Point button in the Toolbar.
![p1](./assets/point_1.png)

Left-click on the image in the Main panel to place a point.
A pop-up dialog will appear, allowing you to name the point. You can either enter a new name or select an used name from the dropdown menu. 

You can view, edit, and delete existing points in the Annotation panel.
![pt1](./assets/place_point_1.png)
<br />

#### **Placing segmentations**

Toggle the Segmentation button in the Toolbar. A Segmentation Toolbar will appear below the main toolbar.
![seg](./assets/seg.png)

Define segmentation classes by selecting the Segmentation tab in the Annotation panel and clicking the Add button.
![seg add](./assets/seg_add.png)

To segment an image, select a segmentation class and activate the Draw button in the Segmentation Toolbar. Left-click and hold to segment, which will be colored based on the color assigned.
![seg 1](./assets/seg_1.png)

For larger regions, draw a closed outline and click Auto Fill in the Segmentation Toolbar to fill the area inside the outline.
![seg 2](./assets/seg_2.png)

Activate the Erase button and erase over the incorrectly segmented area.
<br />

#### **Fast Labelling**

If you often place the same labels across multiple images, PhenoLearn's `Fast Labelling` feature can help speed up the process.

Click `Fast Labelling` in the `Toolbar`. It uses the annotations from the current image as pre-defined annotations for subsequent images. (Note that labels must be placed in the same order to ensure accuracy).
![fast_label](./assets/fast_label.png)
<br />

#### **Saving and Loading Labeling Progress**

You can save your progress at any point by selecting `Save` or `Save As` from the `File menu`. The progress will be saved as a JSON file.

To continue a previous session, use `Open Labelling Progress` from the `File menu` to load a previously saved file.
<br />

#### **Export labels**

Labels from PhenoLearn can be exported via export functions in `File menu` and used as input for Deep Learning models.

PhenoLearn supports exporting annotations in CSV files (both point and segmentation) and black and white masks (only for single-class segmentation).
<br />

### **Use Deep Learning**
This part uses [PhenoTrain](#phenotrain)
-   [Model Training](#model-training)
-   [Model Prediction](#model-prediction)

#### **Model Training**

In the Train tab. There are settings you need to define here:

| Setting | Description |
| --- | --- |
| **Model Type** | `Point` or `Segmentation` <br> [Mask R-CNN](https://arxiv.org/abs/1703.06870) for point <br> [DeepLab](https://arxiv.org/abs/1802.02611) for segmentation. |
| **Input Format** | The default option is `CSV`. The option `Mask` is available when the model type is set to single-class segmentation. |
| **Annotation File** | Annotation file exported from PhenoLearn. |
| **Image Folder** | The folder of the images. |
| **Folder of Black and White Masks** | The folder of the masks when `Input Format` is `Mask` |
| **Image Resize Percentage** | The percentage of resizing the original image <br> Range 1-100% |
| **Validation Set Percentage** | Indicates the percentage of images used in the validation. <br> Default 80/20 |
| **Batch Size** | The batch size for one iteration of training. |
| **Training Epochs** | Training length <br> One epoch is defined as one pass of the full training set|
| **Learning Rate** | Determines how quickly or slowly a neural network model learns a problem. |
| **Training Level** | Both DeepLab and Mask R-CNN use pre-trained parameters. <br> It shows portion of a model is trained on your dataset. Options are Minimal, Intermediate, and Full. |
| **CPU/GPU** | Select whether to use the CPU or GPU for training. If GPU is selected but no GPU is available on the device, the training will automatically default to the CPU.|

Click `Train` to start training. Once completed, a .pth file will be saved in the `saved_model` folder of the PhenoLearnDL root directory.
<br />

The training logs can be viewed in [TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html). They are saved in the `runs` folder. Run the command.

```bash
tensorboard --logdir runs
```

Then visit `http://localhost:6006/` in your browser to view the logs. For detail, you can visit [the tutorial of TensorBoard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html).

#### **Model Prediction**

In the Predict tab. There are settings you need to define here:
<!-- ![Predict view](./assets/predict.png) -->


| Setting | Description |
| --- | --- |
| **Model Type** | `Point` or `Segmentation` |
| **Output Format** | Choose either a PhenoLearn compatible CSV file or black and white masks (only for `Segmentation`). |
| **Model File** | This should be a .pth file saved in training. |
| **Prediction File Folder** | The directory where the prediction file will be saved. |
| **Image Folder** | The directory containing the images for which predictions are to be made. |
| **Name File** | A CSV file with a column 'file' for image names. <br> The name file can be generated by exporting a CSV file PhenoLearn when no annotations are placed. |
| **Image Resize Percentage** | The percentage of resizing the original image <br> Range 1-100% <br>  This should be the same value used in training. |
| **CPU/GPU** | Select whether to use the CPU or GPU for predicting. If GPU is selected but no GPU is available on the device, the predicting will automatically default to the CPU.|

Click `Predict` to start predicting. Once completed, prediction will be saved in `Prediction File Folder`.
<br />

### **Review Predictions**
This part uses [PhenoLabel](#phenolabel)
-   [Import Predictions](#import-predictions)
-   [Review Mode](#review-mode)
-   [Review Assistant](#review-assistant)


#### **Import Predictions**

1. Open an image folder in PhenoLearn, just like in [Image Labeling](#image-labeling).
2. Import your model's predictions into PhenoLearn using the import functions in `File menu`. PhenoLearn can import CSV files and black-and-white masks.
3. After importing, PhenoLearn will visualize the predictions for reviewing


#### **Review Mode**

The `Review Mode` allows users to quickly skim through predictions and flag any inaccuracies.

Activate the `Review Mode` in `Toolbar`. PhenoLearn will display multiple thumbnails of images along with their annotations.

![Review mode](./assets/review_mode.png)


As you skim through the results, tick the checkboxes next to the thumbnails of images with incorrect predictions.

After you've reviewed the images, click the Show Flagged Images button to display only the images you flagged.

![Review mode 2](./assets/review_mode_1.png)

Toggle `Show Flagged Images` in `Toolbar` to only filter the flagged images. And un-toggle `Review Mode` to edit them
<br />

#### **Review Assistant**

The `Review Assistant` allows you to sort and filter images based on their properties.\
Import properties using the `Import Properties` in `File menu`.

- Numerical properties can be used for sorting
- Categorical properties can be used for filtering 

![Review Assistant](./assets/review_assit.png)


#### **Known Issues and Troubleshooting**

##### **1. PyQt5 Installation on macOS**
- **Issue**: Installing PyQt5 could happen on a macOS system may fail.  
- **Solution**:
  1. Install PyQt5 using Homebrew:
     ```bash
     brew install pyqt
     ```
  2. Add PyQt5 to your `PYTHONPATH` in your IDE (e.g., PyCharm).
  3. [Reference Stack Overflow thread](https://stackoverflow.com/questions/76113859/unable-to-install-pyqt5-on-macos) for more details.

##### **2. SSL Certificate Error**
- **Issue**: The error `urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>` could occur when starting training with PhenoTrain on macOS.  
- **Solution**:
  1. Navigate to `/Applications/Python 3.x` (replace `x` with your Python version, e.g., Python 3.10).  
  2. Double-click `Install Certificates.command` to install the SSL certificates.  
  3. Restart your Python environment or IDE and try again.



## License

[MIT](https://opensource.org/licenses/MIT)


