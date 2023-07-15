# PhenoLearn

‘PhenoLearn’ (github.com/EchanHe/PhenoLearn) is an open-source image analysis tool that is designed for high-throughput phenotype measuring on digital biodiversity datasets. It can (i) generates annotations (currently points and segmentation) (ii) use deep learning models to train and predict annotations (iii) Review and edit predictions.

## Table of Contents
-   [Features](#features)
-   [Installation](#installation)
-   [Usage](https://chat.openai.com/?model=gpt-4#usage)
-   [Contributing](https://chat.openai.com/?model=gpt-4#contributing)
-   [License](https://chat.openai.com/?model=gpt-4#license)
  

## Features
- Image Labeling: Explain the functionality and benefits of your image labeling feature here.
- Training Deep Learning Models: Explain the functionality and benefits of your deep learning training feature here.
- Reviewing Deep Learning Predictions: Explain the functionality and benefits of your prediction reviewing feature here.
Add more features as needed
## Installation
Before you start the installation, make sure your system meets the following requirements:

-  Python (Verions >=3.6) installed. If you don't have Python installed, you can get it from [Python](https://www.python.org/downloads/) or [Anaconda Python](https://www.anaconda.com/download). We recommend Anaconda Python, as it has Conda included.
-   pip (Python Package Installer), which typically comes with the Python installation
-   Conda, an open-source package management system and environment management system
  
To install, follow these steps.

<br />
1. Clone the repo
 
```bash
git  clone  https://github.com/EchanHe/PhenoLearn.git
```
It is recommended to create a virtual environment to avoid mixing up the project dependencies with other projects. You can create a virtual environment using the following command:

\
2. Create a new Conda environment and activate it:

```bash
conda create --name phenolearn python
conda activate your_env_name
```
For further information about virtual environment you can visit [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)

\
3. install pacakges:

When you are in the `PhenoLearn` virtual environment, you can install these packages.

Packages required:
- numpy >= 1.20.3
- pandas >= 1.3.4
- opencv-python >= 4.5.5.64
- PyQt >= 5.97

Use the `requirements.txt` in the root directory to install
 ```bash
pip install -r requirements.txt
```

Or just install the newest verions using pip

 ```bash
pip install numpy pandas opencv pyqt
```
\
4. Run PhenoLearn

Navigate to the repository in the terminal and run 
 ```bash
python mainWin.py
```

or Open the repo using [Visual studio Code](https://code.visualstudio.com/)
The run the `mainWin.py` file




## Usage

The window when PhenoLearn starts![Main view](./assets/main.png)

### **Image Labeling**

**Open a project**

Click on the `Open Dir` in the `File menu` to select a directory containing the images to be labeled.
![Main view](./assets/open_1_1.png)


After selecting the directory, PhenoLearn will display all the images in the selected folder in the File panel
![Main view](./assets/opened_1.png)

Click on an image to display it in the Main panel. Here, you can inspect the image and zoom in or out using functions in `View menu`, or use Ctrl mouse wheels. The coordinates and RGB value of your current mouse location are displayed in bottom right

\
**Placing Points**

Activate the Point button in the Toolbar.
![Main view](./assets/point_1.png)

\
Left-click on the image in the Main panel to place a point.
A pop-up dialog will appear, allowing you to name the point. You can either enter a new name or select an used name from the dropdown menu. 

You can view, edit, and delete existing points in the Annotation panel.
![Main view](./assets/place_point_1.png)


**Placing Segmentations**

Toggle the Segmentation button in the Toolbar. A Segmentation Toolbar will appear below the main toolbar.
![Main view](./assets/seg.png)

Define segmentation classes by selecting the Segmentation tab in the Annotation panel and clicking the Add button.
![Main view](./assets/seg_add.png)

To segment an image, select a segmentation class and activate the Draw button in the Segmentation Toolbar. Left-click and hold to segment, which will be colored based on the color assigned.
![Main view](./assets/seg_1.png)

For larger regions, draw a closed outline and click Auto Fill in the Segmentation Toolbar to fill the area inside the outline.
![Main view](./assets/seg_2.png)

Activate the Erase button and erase over the incorrectly segmented area.

\
**Fast Labelling**

If you often place the same labels across multiple images, PhenoLearn's Fast Labelling feature can help speed up the process.

Click Fast Labelling in the Toolbar (Figure 1b).

PhenoLearn will use the annotations from the current image as pre-defined annotations for subsequent images. Note that labels must be placed in the same order to ensure accuracy.

### Training Deep Learning Models

Provide step-by-step instructions on how to use this feature.

### Reviewing Deep Learning Predictions

Provide step-by-step instructions on how to use this feature.

  

### Deep Learning

TODO

## License

[MIT](https://opensource.org/licenses/MIT)

## Contact

If you have any questions, feel free to reach out to us at csyichenhe@Gmail.com.
Enjoy using PhenoLearn! 