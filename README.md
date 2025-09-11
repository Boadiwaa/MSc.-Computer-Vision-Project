## Efficient Robotic Vision in the Field: A Deep Learning Architecture for Strawberry Disease Detection

### Independent Research Project - PBM24 

### Project Structure:

```
📦irp_code                             # main source code directory
 ┣ 📂evaluation_group                  # utilities for evaluation
 ┃  ┗ 📜visualize_n_evaluate.py        # module for main evals and visualization classes and functions  
 ┗ 📂preprocess_group                  # module for preprocessing classes and functions
    ┣ 📜find_remove_duplicates.py      # contains functions to determine duplicate images and filter them out, based on Hamming distance
    ┗ 📜image_processing.py            # contains multiple functions for processing images
📜 requirements.txt                    # contains dependencies for installing the package
📜 LICENSE                             # MIT License for package
📜 environment.yml                     # contains dependencies for installing the package
📜 README.md
📜 pyproject.toml                      # contains dependencies for installing the package
📂models                               # contains model checkpoints for various models. The best models are mvit_upgraded (unsupervised anomaly detector) coupled with mvit_classifier (supervised classifier)
📂notebooks
 ┣ 📜Evaluator.ipynb                   # Jupyter notebook for testing models
 ┣ 📜baseline_model_evals.ipynb        # Jupyter notebook with stats for baseline models
 ┣ 📜mobilevit_train_autoencoder.ipynb # Jupyter notebook with code for training MobileVitV2 as an unsupervised anomaly detector.
 ┣ 📜mvit_classifier.ipynb             # Jupyter notebook with code for training MobileVitV2 in its original state as a classifier.
 ┗ 📜preprocessing_flow.ipynb          # Jupyter notebook with an image preprocessing workflow
📂test_data                            # Contains data for testing models
 ┣ 📂classifier_dataset
 ┣ 📂model_test_patch_b
 ┣ 📂model_test_patch_g
 ┣ 📂patches_bad_only
 ┗ 📂patches_good_only                           
📂title
📂logbook                             # Log of meetings on project
📂deliverables                        # Contains Project Plan and Final report

 ```

### Install and Run
To successfully run the notebooks, you need to import modules stored in the package, `irp_code`. To install `irp_code` here is some guidance:

Due to the ml-cvnets package which houses the MobileViT model backbone, it is recommended to use Python 3.10+ and [PyTorch](https://pytorch.org) (version >= v1.12.0)

Instructions below use Conda. Guidance on installing Conda can be found at: [How to Install Conda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links).

```bash
# For Conda users:
conda env create -f environment.yml
conda activate irp-pbm24

# For pip/ venv users: 
python3.10 -m venv .venv
source .venv/bin/activate #(Windows: venv\Scripts\activate)
python -m pip install -U pip
pip install -r requirements.txt
```

To use CUDA, install the matching Torch CUDA wheels first, then use requirements.txt. Example for CUDA 11.7: 

```bash
pip install --index-url https://download.pytorch.org/whl/cu117 torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 
pip install -r requirements.txt
```

### Data
The test_data directory contains various datasets for testing models. The folders beginning with `model_` have 25 images each and the others have more extensive sets for testing. `classifier_dataset` contains the data for training and testing the MobileVit2 classifier. Some proprietary data has been provided here for successful running of `Evaluator.ipynb` per request of external supervisor, but links to all open-source datasets are provided in the reference section of the final report. 

## Getting started
Run `Evaluator.ipynb` to see the performance of the various models and the final MobileVit2 Anomaly Detector-Classifier ensemble. After activating the created conda environment, from within the Jupyter environment, connect to the kernel associated with the activated environment, and then run the cells.

### License
An MIT License is available for the packaged code `irp_code`, however, the data in `test_data` is predominantly proprietary and thus should not be distributed without written permission from the author and/or Antobot Ltd.