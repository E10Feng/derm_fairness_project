# Assessing Skin Tone Fairness in Dermatology Image Classification
Here is a code framework for training/fine-tuning CNN models on the Fitzpatrick17k image set and then assessing if performance disparities between skin tones can be improved with oversampling and synthetic data augmentation.

## Description
This project was created during the summer of 2024, as part of my internship at Vanderbilt Unviersity's Department of Biomedical Informatics. I worked under the guidance of Nick Jackson and Brad Malin, Ph.D. Here's a link to a detailed abstract for this project: https://docs.google.com/document/d/1mMhQsGaLSjYJF6nK6eZ1HJTuFkaeKqSj93CbP0b_BiI/edit 

## Getting Started
The processed and ready-to-use fitzpatrick17k dataset as well as the synthetic data generated for this project are located here: https://drive.google.com/drive/folders/1OWHdBwBSMSHbGITWz096SQebDhamxRdy?usp=sharing


### Data Processing
Use the included files to generate new image directories & csv files for various levels of oversampling and synthetic augmentation. Remember to change ratio as needed and to copy in original images into the directory after creation.

### Synthetic Data
Synthetic data was generated using StyleGANV3-R (https://github.com/NVlabs/stylegan3) with images resized to 256x256 pixels, a batch size of 32, and gamma=2 using 4XXX GPUs. Training was stopped after the model had 'seen' 15,000,000 images, achieving an FID of 8.73. All other parameters were left as the default parameters of the StyleGAN repo.

### Classifier
Make sure to update all paths to match your set up! 

### Running Experiments
Modify and duplicate the included bash file template to run and store experiments.

### Data Visualization
Modify as needed to generate figures. 

## Help
Email me! 

## Authors
E10 (Ethan) Feng --- e.y.feng@wustl.edu
Nick Jackson --- nicholas.jackson@vanderbilt.edu
