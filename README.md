### FACE SEGMENTATION

#### Data
CelebAMask-HQ Dataset
https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html

#### Metric
Mean Intersection-Over-Union

#### Model 
The following models has been tested in this project: 
 - Unet
 - Unet PlusPlus
 - DeepLabV3
 
 #### Result
 The following results were achieved at Unet:
  
  | Model | Unet | Unet PlusPlus | DeepLabV3 |
  | :---: | :---: | :---: | :---: |
  | mIOU | 0.71 | 0.7 | 0.69 |
  
 #### Installation
 1. Clone repository
 2. `pip install -r requirements.txt`
 3. Set params.json in config dir
 4. `python predict.py <image_path>`
 
 
