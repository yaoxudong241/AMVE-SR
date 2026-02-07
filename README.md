# AMVE-SR
## Activation Map-Based Visual Explanation for Remote Sensing Image Super-Resolution

## Environment
- python 3.8
- pytorch=2.1.0
- numpy=1.24.3
- rasterio=1.3.10

## Model 

![1](IMG/SRCAMmethod.png)
Pipeline of AMVE-SR. AMVE-SR computes output-layer gradients using the edge loss of the patch to be explained, generates activation maps, and propagates them layer by layer back to the input.

## Train 
The model is pre-trained on the AID dataset and is saved in the "weights" folder.

## Interpretation Map Generation
```sh
python main_EDSR.py --HRPath /path/to/hrimg/ --LRPath /path/to/lrimg/ --savePath /path/to/interpretation maps/ --checkpoint /path/to/model/
```
## Quantitative evaluation

```sh
python edgeloss.py
```
The corresponding path needs to be adjusted.


## Visual comparison
![1](IMG/SRCAM.png)
The visualization of the interpretation results in different remote sensing scenes includes the activation maps for the output layer, hidden layer, and input layer.

## Results
More comparison results can be found at  [Google Drive](https://drive.google.com/drive/folders/1D3l540x9emPR_oy2yc9rpISbAJmTNwV5?dmr=1&ec=wgc-drive-hero-goto?usp=drive_link). 
The results corresponding to different methods are stored in separate folders. The folder based on the CAM method contains interpretation maps of the input layer, output layer, and hidden layers, as well as the overlays of these interpretation maps with the original images.






