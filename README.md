# Mask R-CNN

This repository contains an implementation of Mask R-CNN, a state-of-the-art model for instance segmentation and object detection. Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression.

## Project Structure

The repository is organized as follows:

- `.gitignore`: Specifies files and directories to be ignored by Git.
- `CIS_680_Final_report.pdf`: A comprehensive report detailing the project, including methodology, experiments, and results.
- `boxhead_infer.py`: Script for performing inference on bounding box heads.
- `datasets/`: Directory containing dataset utilities and dataset-related scripts.
- `mask_head_infer.py`: Script for performing inference on mask heads.
- `model/`: Contains model definitions and related files.
- `pics/`: Directory for storing images and visualizations.
- `rpn_infer.py`: Script for performing inference on Region Proposal Networks (RPNs).
- `trainer/`: Contains training scripts and modules.
- `utils/`: Contains utility functions and scripts for common operations.

## Requirements

The project requires the following dependencies:

- Python 3.8 or higher
- PyTorch
- torchvision
- NumPy
- OpenCV
- Matplotlib
- Any additional requirements specified in the `requirements.txt` (if provided)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MaskRCNN.git
   cd MaskRCNN
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:
   - Place your dataset in the `datasets/` directory.
   - Update dataset paths in the relevant scripts.

## Usage

### Training

To train the Mask R-CNN model, navigate to the `trainer/` directory and run the training script:
```bash
python train.py 
```
Ensure is correctly set up with dataset paths, hyperparameters, and other settings.

### Inference

1. **Bounding Box Head Inference:**
   ```bash
   python boxhead_infer.py 
   ```

2. **Mask Head Inference:**
   ```bash
   python mask_head_infer.py 
   ```

3. **Region Proposal Network Inference:**
   ```bash
   python rpn_infer.py 
   ```

## Results and Visualizations

The `pics/` directory contains visualizations of results, including segmented images, bounding boxes, and masks. You can save new results here by specifying the output paths in the inference scripts.

## Contributions

Contributions to this repository are welcome. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of the changes.


## Acknowledgements

- This implementation is inspired by the Mask R-CNN paper by Kaiming He et al.
- Special thanks to the contributors and maintainers of the libraries used in this project.

## Contact

For questions or issues, please open an issue on the repository or contact the project maintainer.
