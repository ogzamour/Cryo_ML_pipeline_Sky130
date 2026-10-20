# Cryo_ML_pipeline_Sky130

NGSpice simulation using translated cryogenic PDk models based on https://github.com/UTA-Advanced-Detector-Technologies/Skywater-130nm-77K-Cryogenic-Models/tree/main 

***download this repo to ngspice-skywater-sims directory in your home directory***
## Downloading_Volare

First, ensure Volare is properly set up (I did this by creating a seperate file titled skywater130nm where I placed all the things downloaded from volare (which should be 3 top files: volare, sky130A, sky130B.) 
My path essentially looked like /home/<user>/skywater130nm/volare...)

To download volare, I used the following (with <desired_PDK_directory> = skywater130nm):

```
#Taken from original cryo repo

pip install volare

export PDK_ROOT="/home/<your_username>/<desired_PDK_directory>"

volare ls-remote --pdk sky130

volare enable --pdk sky130 a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b
```
## Downloading_Ngspice

I used jupyter notebooks for this, so that is the setup I will show. However, just make sure version 41 of ngspice is installed and it should work.
This is the setup I used, and I just made sure sims were run in spice_env:
```
  conda activate spice_env
  conda install -c conda-forge ngspice=41
  which ngspice
  ngspice --version
```

## Enviornment_Setup
In order to correctly set up the 77k nfet and pfet files to work with the rest of volare, you will want to place the updated PDKs as well as the prevMeth PDKs (found in the main directory of the repo) for both device types in the following directory:
```
skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.ref/sky130_fd_pr/spice
```
Once moved, if running monte carlo simulation, delete the ```updated_``` suffix from both file names. If running ngSpice comparisons, delete the ```prevMeth_``` suffix from both file names.
***ensure only one of these PDK types (either updated or prevMeth) has its suffix removed at a time ***

Now, you will want to update the sky130.lib.spice file (/home/<user>/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.tech/ngspice/sky130.lib.spice)
To do this, insert the following code in sky130.lib.spice under '******* SkyWater sky130 model library *********':

```
* Typical corner (tt) at 77K
.lib tt_77k
.param mc_mm_switch=0
.param mc_pr_switch=0
.param my_gauss = 0
.param m = 1
.include "/home/<user>/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__nfet_01v8_lvt__tt_77k.corner.spice"
.include "/home/<user>/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.ref/sky130_fd_pr/spice/pfet_01v8_lvt/77k_models/sky130_fd_pr__pfet_01v8_lvt__tt_77k.corner.spice"

.include "/home/<user>/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.ref/sky130_fd_pr/spice/sky130_fd_pr__nfet_01v8_lvt__mismatch.corner.spice"
.include "/home/<user>/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.ref/sky130_fd_pr/spice/pfet_01v8_lvt/sky130_fd_pr__pfet_01v8_lvt__mismatch.corner.spice"

.include "r+c/res_typical__cap_typical.spice"
.include "r+c/res_typical__cap_typical__lin.spice"

* Special cells
.include "corners/tt/specialized_cells.spice"


.endl
```
After this, to use the pdk at 77k, just ensure the .lib declaration in the ngspice simulation files includes tt_77k rather than tt!

## Verifying NgSpice Viability
***ensure paths in each file match the paths in your directory***

In order to verify the accuracy of our simulation tool, we recreated ran a VG and VD sweeps for each device at several bias voltages to match the data shown in the reference paper. Ensure the ```prevMeth``` PDKs are being used (see Enviormnent_Setup).
The ngspice simulations can be found inside the ```pfet_mod``` and ```nfet_mod``` folders. Inside each folder, ```runAll.py``` can be ran to simulate all the necessary data for each transistor type. 

Single device simulations can be run by typing ```ngspice -b <desiredSimFile>.spice``` in the terminal while in the simulation directory. 

To view plotted ngSpice simulations against the measurements for each device, see ```comp_analysis``` folder and run the notebook ```spice_analysis_plots.ipynb``` for the desired doping type and device after simulating the data.

***(if you set ngspice in a specific enviornment like spice_env, make sure it is active while trying to run the simulation)***


## RRMS

To get the RRMS calculation for the baseline model scripts, first ensure you have the cryo_data file from the UTA repo downloaded. Also ensure both the nfet and pfet 77k corners are installed properly and defined in the .lib file.
Then, run ```python runAll.py``` in both the nfet_mod and pfet_mod directories ***Note: will have to change path in the .include***

After both are run, ensure the paths in getRMS.py match your directory (for both the cryo data and modeled pfet/nfet data). Then, run ```python rrmsCalc.py``` in the RRMS directory and the RRMS should print in terminal.

***could also choose to not rerun pfet/nfet models since the output files are included already in repo***


## Training Data Generation
To generate parameter training data for a single device in the PDK, run the notebook ```latinHypercube_DataGen.ipynb``` in the montecarlo directory. The sampling size chan be changed bye editing the ```n``` variable, and the device for the sweep can be edited as well by observing the device library. s
The output will be in ```mc_output_lhc``` and all sweep data will be in 1 CSV files per parameter vg sweep. (Option to save vd sweeps can also be set, and bias voltage can be changed)
***Currently, everything is set for a vg sweep only over NMOS model 0 (0.15 x 1.6) at 1.48 bias voltage. This can all be changed in the code for other models, but several files will need to be updated as well as the parameter ranges for the LHC sweep.***

## Machine-Learning Model for Stage01 (Low Threshold Voltage Parameters)
After sufficient training data has been generated for a desired model, an MLP can now be trained to predict parameter extraction. 
All scripts for the model training can be found in the folder ```ml_method1```. Again, first ensure your input and output libraries in each script reflect your directory. 

First, run the ```ml_00_data_analysis.ipynb``` notebook. This will format the training data in a way the network can use. Make sure the number of simulations ```n``` reflects the amount of training data you generated. 

After the data has been reformatted, the surrogate model can be made by first running the ```surrogate_train.ipynb``` and then running ```surrogate_predict.ipynb``` to view the results. 
Once the surrogate has been created, the main model can be made by running ```ml_01_train_keras.ipynb```, which has several variables you can mess with (these can also be found in the ```parameters.py``` file). Finally, you can view the models prediction by running ```ml_02_predict_using_measurements.ipynb```, and once the new parameters have been sent to an ngSpice file, run ```ngspice -b testParam.spice``` in the terminal and continue running the notebook to view the comparison plots.

***There are two different network library options in each notebook, pyTorch and Keras***

***currently the ml is configured for a single device at a single bias vd. This can be altered if you wish. There are also some supplementary scripts throughout the directory that are not 
