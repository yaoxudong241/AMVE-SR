# SRCAM
## Exploring Potential of Class Activation Maps for Interpreting Super-Resolution Networks

## Environment
- python 3.8
- pytorch=2.1.0
- numpy=1.24.3
- rasterio=1.3.10

## Model 

![1](IMG/SRCAMmethod.png)
The pipeline of SRCAM. SRCAM begins with the calculation of edge loss between the SR and HR patches to be interpreted. Subsequently, the gradients derived from the edge loss are utilized to generate the activation map for the output layer. Finally, this activation map is propagated layer by layer back to the input layer.

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
More comparison results can be found in the "data" folder. The results corresponding to different methods are stored in separate folders. The folder based on the CAM method contains interpretation maps of the input layer, output layer, and hidden layers, as well as the overlays of these interpretation maps with the original images.


