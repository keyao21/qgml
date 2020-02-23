# ML_Fluid 

This project implements a machine learning algorithm to calibrate models on stream function values from two fluid models, double gyre and quasi-geostrophic. 

## Demo 

### Double Gyre Experiment
TODO: I think there might be something wrong with the ESN outputs from the double gyre model? debug test.py, specifically the outputs/attributes of esn after calling esn.test() 

1. Run double_gyre.py with experiment parameters (script is located in __~/QG_FTLE/src/double_gyre.py__). A stream function time series data file will be saved down in __~/QG_FTLE/inputs__. 
2. Copy the stream function time series data file from __~/QG_FTLE/inputs__ to __~\ML_Fluid\inputs\raw__
3. Run ```dg_demo.py```, make sure that ```configs.py``` has the proper set up. 


### Quasigeo Experiment
Currently running an experiment to check model outputs' dependency on reservoir size. Varying resSize param (250,750,1000). 
TODO: need to add .mat file to __~/ML_Fluid\inputs\raw__ in order to process and train models on stream function. 
TODO: CHeck model dependency on spectral radius
