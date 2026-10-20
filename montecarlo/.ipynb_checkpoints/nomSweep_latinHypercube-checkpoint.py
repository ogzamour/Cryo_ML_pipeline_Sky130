import numpy as np
import matplotlib.pyplot as plt
import subprocess
import csv
import os
from datetime import datetime
from pathlib import Path
import time
from nominal_lib import *
from getCSV import makeCSV
from getIV import getIvVd, getIvVg
from scipy.stats import qmc
import re
N_RUNS = 10000
#SPICE_FILE = "nfet_base.spice"
PARENT_DIR = Path.home() / "ngspice-skywater-sims/montecarlo/mc_output_lhc"
desired_device = 0
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
device_library = [("nmos_FET_len_0p15_wid_1p6", model_devs[0],'n', '0'),
           ("nmos_FET_len_0p19_wid_7", model_devs[1],'n', '1'),
           ("nmos_FET_len_0p25_wid_1p6", model_devs[2],'n','2'),
           ('nmos_FET_len_1_wid_1p6', model_devs[3],'n','3'),
           ('nmos_FET_len_1_wid_3', model_devs[4],'n', '4'),
           ('nmos_FET_len_8_wid_1p6', model_devs[5],'n', '5'),
           ('nmos_FET_len_20_wid_0p64', model_devs[6],'n','6'),
           ('nmos_FET_len_100_wid_100', model_devs[7],'n','7'),
           ('pmos_FET_len_8_wid_0p84', model_devs[8],'p','8'),
           ('pmos_FET_len_0p35_wid_1p6', model_devs[9],'p','0'),
           ('pmos_FET_len_0p5_wid_0p42', model_devs[10],'p','7'),
           ('pmos_FET_len_0p35_wid_0p55', model_devs[11],'p','5'),
           ('pmos_FET_len_0p5_wid_0p64', model_devs[12],'p','4'),
           ('pmos_FET_len_0p35_wid_5', model_devs[13],'p','9'),
           ('pmos_FET_len_2_wid_5', model_devs[14],'p','11'),
           ('pmos_FET_len_4_wid_7', model_devs[15],'p','2'),
           ('pmos_FET_len_8_wid_1p6', model_devs[16],'p','6'),
           ('pmos_FET_len_8_wid_5', model_devs[17],'p','1')
]
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getHypercube(num_samples, runDir, p):
    num_vars = 7 # e.g., Width (W) and Length (L)
    # 1. Define exact parameter order to match columns
    PARAMS = ["vth0", "vsat", "rdsw", "u0", "nfactor", "eta0", "delta"]
    #num_vars = len(PARAMS)
    
    # 2. Your specified multipliers/ranges, calculated using paramRangeValidation
    n_bounds = {
        "vth0": (0.1, 10),
        "vsat": (0.1, 10),
        "rdsw": (0.2, 10),
        "u0": (0.6905, 10),
        "nfactor": (0.1, 50),
        "eta0": (0.1, 10),
        "delta": (0.0003, 10),
    }
    
    # 3. Generate LHS sample matrix in [0, 1] space
    #num_samples = 500  # or your desired n
    sampler = qmc.LatinHypercube(d=num_vars)
    sample_matrix = sampler.random(n=num_samples)
    
    # 4. Extract floors and ceilings in the exact order of PARAMS
    floor = np.array([n_bounds[p][0] for p in PARAMS])
    ceiling = np.array([n_bounds[p][1] for p in PARAMS])
    
    # 5. Scale from [0, 1] to your custom bounds
    scaled_matrix = qmc.scale(sample_matrix, floor, ceiling)
    
    # 6. Export to a text file for ngspice to read
    #os.makedirs(runDir, exist_ok=True)
    '''np.savetxt(
        os.path.join(runDir, "lhs_parameters.txt"), scaled_matrix, fmt="%.16e"
    )
    print(f"Generated LHS parameter matrix of shape: {scaled_matrix.shape}")'''
        
    # Export to a text file for ngspice to read
    np.savetxt(os.path.join(runDir,'lhs_parameters.txt'), scaled_matrix, fmt='%.16e')
   
    return
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getOldHypercube(num_samples, runDir, p):
    num_vars = 7 # e.g., Width (W) and Length (L)
    #p = np.loadtxt(os.path.join(dev_dir, "model_parameters.txt")) #original parameters for setting bounds
    
    sampler = qmc.LatinHypercube(d=num_vars)
    sample_matrix = sampler.random(n=num_samples)

    b1 = 0.8 * p
    b2 = 2 * p
    floor = np.minimum(b1, b2)   # Forces the truest minimum
    ceiling = np.maximum(b1, b2)

    scaled_matrix = qmc.scale(sample_matrix, floor, ceiling)
    
    # Export to a text file for ngspice to read
    np.savetxt(os.path.join(runDir,'lhs_parameters.txt'), scaled_matrix, fmt='%.16e')
   
    return
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getHypercubeFrom300K(num_samples, runDir, param_300k, devType):
    param_list = ["vth0", "vsat", "rdsw", "u0", "nfactor", "eta0", "delta"]
    num_vars = len(param_list)
    
    n_bounds = {
        "vth0": (0.8, 1.6),
        "vsat": (0.8, 2.5),
        "rdsw": (0.3, 1.5),
        "u0":   (1.50, 3.50),
        "nfactor": (0.80, 50), 
        "eta0": (0.90, 1.20),
        "delta": (0.80, 1.50)
    }
    p_bounds = {
        "vth0": (0.8, 1.6),
        "vsat": (1.05, 1.35),
        "rdsw": (0.50, 0.95),
        "u0":   (1.50, 3.50),
        "nfactor": (0.80, 1.50), 
        "eta0": (0.90, 1.20),
        "delta": (0.80, 1.50)
    }
    if devType == 'n':
        bounds = n_bounds
    else:
        bounds = p_bounds
    
    lower_bounds = [bounds[param][0] for param in param_list]
    upper_bounds = [bounds[param][1] for param in param_list]
    
    sampler = qmc.LatinHypercube(d=num_vars)
    sample_matrix = sampler.random(n=num_samples)

    # Scale to get the raw multiplying factors [0, 1] mapped to bounds
    multiplier_matrix = qmc.scale(sample_matrix, np.array(lower_bounds), np.array(upper_bounds))
    
    # Compute absolute parameter values using the exact same param_list order
    
    os.makedirs(runDir, exist_ok=True)
    np.savetxt(os.path.join(runDir, 'lhs_parameters.txt'), multiplier_matrix, fmt='%.16e')
   
    return 
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def roomTempParams(devType, dev_param, SPICE_FILE):
    home = str(Path.home())
    if devType == 'p':
        biasV = -1.48
    else:
        biasV = 1.48
    device_cmd = f"""* Room temp sweep (Sky130) 


{dev_param} nf=1 ad='floor((nf+1)/2) * w/nf * 0.29' as='floor((nf+2)/2) * w/nf * 0.29' pd='2 * floor((nf+1)/2) * (w/nf +0.29)' ps='2 * floor((nf+2)/2) * (w/nf + 0.29)' nrd='0.29 / w' nrs='0.29 / w' sa=0 sb=0 sd=0 

.lib "{home}/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.tech/ngspice/sky130.lib.spice" tt_77k

Vgs gate GND {biasV}
Vds VDD GND {biasV}

"""
    control_lines = [
    ".control",
    "set noaskquit",
    "set filetype=ascii",
    "run",
]
    
    control_lines.extend([
    
    # Dynamically update the specific instance parameters
    f'alterparam vth0_nom = 1',
    f'alterparam u0_nom = 1',
    f'alterparam rdsw_nom = 1',
    f'alterparam nfactor_nom = 1',
    f'alterparam vsat_nom = 1',
    f'alterparam eta0_nom = 1',
    f'alterparam delta_nom = 1',
    'reset',
    'dc Vgs 0 1.85 0.01',
    'showmod m.xm1.msky130_fd_pr__nfet_01v8_lvt : vth0 u0 rdsw nfactor vsat eta0 delta > roomTempParam.txt',
    'let ax_id2 = -vds#branch',
    f'wrdata roomTemp_vgs.dat vdd gate ax_id2'
        ])
    control_lines.extend(["quit", ".endc", ".end"])
    with open("mc_run.spice", "w+") as f:
    # Pre-read the base template so you don't read it from disk over and over
        #base_template = open(SPICE_FILE).read()
        print ('opened')
        f.seek(0)    
        f.truncate()  
        f.write(device_cmd)
        f.write("\n\n")
        f.write("\n".join(control_lines))
        f.flush()     


    logFile = os.path.join(run_dir, "mc_run.log")
    with open(logFile, 'w') as log:
        subprocess.run(["ngspice", "-b","-q", "mc_run.spice"], stdout=log, stderr=subprocess.STDOUT)#, capture_output=True, text=True)

   
    return
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def lhcVisual(runDir):
    labels = ["vth0", "vsat", "rdsw", "u0", "nfactor", "eta0", "delta"]
    
    file_path = os.path.join(runDir, 'lhs_parameters.txt')
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    matrix = np.loadtxt(file_path)
    num_samples, num_vars = matrix.shape
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(14, 14), sharex='col')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    for i in range(num_vars):
        for j in range(num_vars):
            ax = axes[i, j]       
            #corner grid layout
            if j > i:
                ax.set_visible(False)
                continue    
            #histogram on diagonal
            if i == j:
                ax.hist(matrix[:, i], bins=12, color=colors[i], edgecolor='white', alpha=0.8)
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)
            #plotting multivariable scatter representations
            else:
                ax.scatter(matrix[:, j], matrix[:, i], color='#555555', alpha=0.5, edgecolors='none', s=20)
                ax.set_ylim(matrix[:, i].min() * 0.98, matrix[:, i].max() * 1.02)
            #boundaries and layout frames
            ax.set_xlim(matrix[:, j].min() * 0.98, matrix[:, j].max() * 1.02)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #format peripheral label text on edge borders only
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=11, fontweight='bold')
            if i == num_vars - 1:
                ax.set_xlabel(labels[j], fontsize=11, fontweight='bold')
                ax.tick_params(axis='x', labelrotation=45)
    #saving
    fig.tight_layout()
    outdir = os.path.join(runDir, "allPlots")
    outfile = os.path.join(outdir, "LHC_Sampling_Visualization.png")
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    
    print(f"Saved {outfile}")
    plt.close(fig)
    return
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getRoomTempValues():
    bsim_params = {}


    pattern = re.compile(
        r"\b([a-zA-Z0-9_]+)\s*=\s*([+-]?\d*\.?\d+(?:[eE][+-]?[0-9]+)?)"
        r"|\b([a-zA-Z0-9_]+)\s+([+-]?\d*\.?\d+(?:[eE][+-]?[0-9]+)?)"
    )
    file_path = "roomtempparam.txt"
    
    with open(file_path, "r") as file:
      for line in file:
        # Find all matches in the current line
        matches = re.findall(
            r"([a-zA-Z0-9_]+)\s+([+-]?\d*\.?\d+(?:[eE][+-]?[0-9]+)?)", line
        )
        for key, val in matches:
          bsim_params[key.lower()] = float(val)
    
    # Display the extracted parameters
    #print("Extracted BSIM4 Parameters:")
    #for param, value in bsim_params.items():
      #print(f"{param}: {value}")
    return bsim_params

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def deviceTemplate(dev_dir, dev_param, devType):
    if devType == 'p':
        biasV = -1.48
    else:
        biasV = 1.48
    home = str(Path.home())
    #nfet_meas = 
    
    device_cmd = f"""* NMOS sweep (Sky130) 


{dev_param} nf=1 ad='floor((nf+1)/2) * w/nf * 0.29' as='floor((nf+2)/2) * w/nf * 0.29' pd='2 * floor((nf+1)/2) * (w/nf +0.29)' ps='2 * floor((nf+2)/2) * (w/nf + 0.29)' nrd='0.29 / w' nrs='0.29 / w' sa=0 sb=0 sd=0 

.lib "{home}/skywater130nm/volare/sky130/versions/a918dc7c8e474a99b68c85eb3546b4ed91fe9e7b/sky130A/libs.tech/ngspice/sky130.lib.spice" tt_77k

Vgs gate GND {biasV}
Vds VDD GND {biasV}
.temp -196.15


"""
    file = 'mc_template.spice'
    outFile = os.path.join(dev_dir, file)
    if not os.path.exists(outFile):
        with open(outFile, "w+") as f:
            f.write(device_cmd)
            f.flush()
        
    return outFile
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
start_time = time.perf_counter()
folder = datetime.now().strftime(f"{N_RUNS}_runs_data_%Y-%m-%d_%H-%M-%S")

# device setup
device = device_library[desired_device]
deviceName = device[0]
circuit = device[1]
fet = device[2]
nom_params_key = f"{fet}fet_mod_{device[3]}"
modP = mods_list.get(nom_params_key, {})

# directory + template setup
dev_dir = PARENT_DIR / deviceName
run_dir = dev_dir / folder
run_dir.mkdir(parents=True, exist_ok=True)
print(run_dir)
plotDir = run_dir / "allPlots"
plotDir.mkdir(parents=True, exist_ok=True)
deviceTemplate(dev_dir, circuit, fet)
SPICE_FILE = os.path.join(dev_dir, "mc_template.spice")

# boundary setup and LHC run (Generates all parameters at once)
param_init = np.array([
    float(modP["vth0_nom"]), 
    float(modP["vsat_nom"]), 
    float(modP["rdsw_nom"]), 
    float(modP["u0_nom"]), 
    float(modP["nfactor_nom"]), 
    float(modP["eta0_nom"]), 
    float(modP["delta_nom"])
])

getHypercube(N_RUNS, run_dir, param_init)
lhcVisual(run_dir)
lhs_runs = np.loadtxt(os.path.join(run_dir, "lhs_parameters.txt"))

# Base template configuration
base_template = open(SPICE_FILE).read()
logFile = os.path.join(run_dir, "mc_run.log")

CHUNK_SIZE = 10000
total_runs = len(lhs_runs)

# Process runs in increments of 10,000
for chunk_start in range(0, total_runs, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, total_runs)
    chunk_runs = lhs_runs[chunk_start:chunk_end]
    
    control_lines = [
        ".control",
        "set noaskquit",
        "set filetype=ascii",
        "run",
    ]
    
    # Build control lines keeping the absolute global index (idx) for file naming
    for offset, run in enumerate(chunk_runs):
        idx = chunk_start + offset
        v_vth0, v_vsat, v_rdsw, v_u0, v_nfactor, v_eta0, v_delta = run
        runPath = Path(run_dir)
        outPath = runPath
        vds_file = outPath / f"vds_sweep_{idx}.dat"
        vgs_file = outPath / f"vgs_sweep_{idx}.dat"

        if fet == 'n':
            control_lines.extend([
                f'echo " MONTE CARLO RUN: {idx}"',
                f'alterparam vth0_nom = {v_vth0}',
                f'alterparam u0_nom = {v_u0}',
                f'alterparam rdsw_nom = {v_rdsw}',
                f'alterparam nfactor_nom = {v_nfactor}',
                f'alterparam vsat_nom = {v_vsat}',
                f'alterparam eta0_nom = {v_eta0}',
                f'alterparam delta_nom = {v_delta}',
                'reset',
                #'alter Vgs 1.48',
                #'dc Vds 0 1.85 0.01',
                #f'let nom_vth0    = {v_vth0}',
                #f'let nom_u0      = {v_u0}',
                #f'let nom_rdsw    = {v_rdsw}',
                #f'let nom_nfactor = {v_nfactor}',
                #f'let nom_vsat    = {v_vsat}',
                #f'let nom_eta0    = {v_eta0}',
                #f'let nom_delta   = {v_delta}',
                #f'let run_num     = {idx}',
                #'let ax_id0 = -vds#branch',
                #'set wr_singlescale',
                #f'wrdata {vds_file} vdd gate ax_id0 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
                'dc Vgs 0 1.85 0.01',
                f'let nom_vth0    = {v_vth0}',
                f'let nom_u0      = {v_u0}',
                f'let nom_rdsw    = {v_rdsw}',
                f'let nom_nfactor = {v_nfactor}',
                f'let nom_vsat    = {v_vsat}',
                f'let nom_eta0    = {v_eta0}',
                f'let nom_delta   = {v_delta}',
                f'let run_num     = {idx}',
                'let ax_id2 = -vds#branch',
                'set wr_singlescale',
                f'wrdata {vgs_file} vdd gate ax_id2 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta'
            ])
        else:
            control_lines.extend([
                f'echo " MONTE CARLO RUN: {idx}"',
                f'alterparam vth0_nom = {v_vth0}',
                f'alterparam u0_nom = {v_u0}',
                f'alterparam rdsw_nom = {v_rdsw}',
                f'alterparam nfactor_nom = {v_nfactor}',
                f'alterparam vsat_nom = {v_vsat}',
                f'alterparam eta0_nom = {v_eta0}',
                f'alterparam delta_nom = {v_delta}',
                'dc Vds 0 -1.85 -0.01',
                f'let nom_vth0    = {v_vth0} * (v(gate)*0 + 1)',
                f'let nom_u0      = {v_u0} * (v(gate)*0 + 1)',
                f'let nom_rdsw    = {v_rdsw} * (v(gate)*0 + 1)',
                f'let nom_nfactor = {v_nfactor} * (v(gate)*0 + 1)',
                f'let nom_vsat    = {v_vsat} * (v(gate)*0 + 1)',
                f'let nom_eta0    = {v_eta0} * (v(gate)*0 + 1)',
                f'let nom_delta   = {v_delta} * (v(gate)*0 + 1)',
                f'let run_num     = {idx} * (v(gate)*0 + 1)',
                'let ax_id0 = -vds#branch',
                'set wr_singlescale',
                f'wrdata {vds_file} vdd gate ax_id0 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
                'dc Vgs 0 -1.85 -0.01',
                f'let nom_vth0    = {v_vth0} * (v(gate)*0 + 1)',
                f'let nom_u0      = {v_u0} * (v(gate)*0 + 1)',
                f'let nom_rdsw    = {v_rdsw} * (v(gate)*0 + 1)',
                f'let nom_nfactor = {v_nfactor} * (v(gate)*0 + 1)',
                f'let nom_vsat    = {v_vsat} * (v(gate)*0 + 1)',
                f'let nom_eta0    = {v_eta0} * (v(gate)*0 + 1)',
                f'let nom_delta   = {v_delta} * (v(gate)*0 + 1)',
                f'let run_num     = {idx} * (v(gate)*0 + 1)',
                'let ax_id2 = -vds#branch',
                f'wrdata {vgs_file} vdd gate ax_id2 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
                'reset'
            ])

    control_lines.extend(["quit", ".endc", ".end"])

    # Write chunk-specific spice deck
    with open("mc_run.spice", "w+") as f:
        f.write(base_template)
        f.write("\n\n")
        f.write("\n".join(control_lines))
        f.flush()

    # Execute ngspice for this batch
    mode = 'a' if chunk_start > 0 else 'w'
    with open(logFile, mode) as log:
        subprocess.run(["ngspice", "-b", "-q", "mc_run.spice"], stdout=log, stderr=subprocess.STDOUT)

    print(f"Finished sweeps for runs {chunk_start} to {chunk_end - 1}.")

end_time = time.perf_counter()
duration = end_time - start_time
print(f"Task completed in {duration:.4f} seconds.")
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''start_time = time.perf_counter()
folder = datetime.now().strftime(f"{N_RUNS}_runs_data_%Y-%m-%d_%H-%M-%S")

#device setup
device = device_library[desired_device]
deviceName = device[0]
circuit = device[1]
fet = device[2]
nom_params_key = f"{fet}fet_mod_{device[3]}"
modP = mods_list.get(nom_params_key, {})

#directroy + template setup
dev_dir = PARENT_DIR / deviceName
run_dir = dev_dir / folder
run_dir.mkdir(parents=True, exist_ok=True)
print(run_dir)
plotDir = run_dir / "allPlots"
plotDir.mkdir(parents=True, exist_ok=True)
deviceTemplate(dev_dir, circuit, fet)
SPICE_FILE = os.path.join(dev_dir, "mc_template.spice")

#roomTempParams(fet, circuit, SPICE_FILE)
#roomTemp_params = getRoomTempValues()

param_init_300k = np.array([
    float(roomTemp_params[0]), 
    float(roomTemp_params[1]), 
    float(roomTemp_params[2]), 
    float(roomTemp_params[3]), 
    float(roomTemp_params[4]), 
    float(roomTemp_params[5]), 
    float(roomTemp_params[6])
])
param_init_300k = {
    "vth0" : float(roomTemp_params[0]), 
    "vsat" : float(roomTemp_params[4]), 
    "rdsw" : float(roomTemp_params[2]), 
    "u0" : float(roomTemp_params[1]), 
    "nfactor" : float(roomTemp_params[3]), 
    "eta0" : float(roomTemp_params[5]), 
    "delta" : float(roomTemp_params[6])
}
#boundary setup and LHC run
param_init = np.array([
    float(modP["vth0_nom"]), 
    float(modP["vsat_nom"]), 
    float(modP["rdsw_nom"]), 
    float(modP["u0_nom"]), 
    float(modP["nfactor_nom"]), 
    float(modP["eta0_nom"]), 
    float(modP["delta_nom"])
])

getHypercube(N_RUNS, run_dir, param_init)

lhcVisual(run_dir)
lhs_runs = np.loadtxt(os.path.join(run_dir, "lhs_parameters.txt"))
#vth0_v, u0_v, rdsw_v, nfactor_v, vsat_v, eta0_v, delta_v = lhs_runs.T

#netlist setup for each LHC sweep
control_lines = [
    ".control",
    "set noaskquit",
    "set filetype=csv",
    "run",

]
for idx, run in enumerate(lhs_runs):
    v_vth0, v_vsat, v_rdsw, v_u0, v_nfactor, v_eta0, v_delta = run
    runPath = Path(run_dir)
    #outdir = f"run_{idx}"
    #outPath = runPath / outdir
    #outPath.mkdir(parents=True, exist_ok=True)
    outPath = runPath
    vds_file = outPath / f"vds_sweep_{idx}.dat"
    #vgs_single_file = outPath / "vgs_sweep_0p01_vd.csv"
    vgs_file = outPath / f"vgs_sweep_{idx}.dat"
    p_file = outPath / "new_params.txt"

    if fet == 'n':
        control_lines.extend([
    f'echo " MONTE CARLO RUN: {idx}"',
    # Dynamically update the specific instance parameters
    f'alterparam vth0_nom = {v_vth0}',
    f'alterparam u0_nom = {v_u0}',
    f'alterparam rdsw_nom = {v_rdsw}',
    f'alterparam nfactor_nom = {v_nfactor}',
    f'alterparam vsat_nom = {v_vsat}',
    f'alterparam eta0_nom = {v_eta0}',
    f'alterparam delta_nom = {v_delta}',
    'reset',
    
    #'dc Vds 0 1.85 0.01 Vgs 0 1.85 0.185',
    'alter Vgs 1.48',
    'dc Vds 0 1.85 0.01',
    #f'show m.xm1.msky130_fd_pr__nfet_01v8_lvt | tee {p_file} > /dev/null',
    #f'wrdata {p_file} vth0 u0 rdsw nfactor vsat eta0 delta',
    f'let nom_vth0    = {v_vth0}',
    f'let nom_u0      = {v_u0}',
    f'let nom_rdsw    = {v_rdsw}',
    f'let nom_nfactor = {v_nfactor}',
    f'let nom_vsat    = {v_vsat}',
    f'let nom_eta0    = {v_eta0}',
    f'let nom_delta   = {v_delta}',
    f'let run_num     = {idx}',
    'let ax_id0 = -vds#branch',
    #'set wr_vecnames',
    'set wr_singlescale',
    f'wrdata {vds_file} vdd gate ax_id0 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
    
    #'dc Vgs 0 1.85 0.01 Vds 0.01 0.01 1',
    #f'let nom_vth0    = {v_vth0} * (v(gate)*0 + 1)',
    #f'let nom_u0      = {v_u0} * (v(gate)*0 + 1)',
    #f'let nom_rdsw    = {v_rdsw} * (v(gate)*0 + 1)',
    #f'let nom_nfactor = {v_nfactor} * (v(gate)*0 + 1)',
    #f'let nom_vsat    = {v_vsat} * (v(gate)*0 + 1)',
    #f'let nom_eta0    = {v_eta0} * (v(gate)*0 + 1)',
    #f'let nom_delta   = {v_delta} * (v(gate)*0 + 1)',
    #f'let run_num     = {idx} * (v(gate)*0 + 1)',
    #'let ax_id1 = -vds#branch',
    #f'wrdata {vgs_single_file} vdd gate ax_id1 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
    
    #'dc Vgs 0 1.85 0.01 Vds 0 1.85 0.185',
    'dc Vgs 0 1.85 0.01',
    f'let nom_vth0    = {v_vth0}',
    f'let nom_u0      = {v_u0}',
    f'let nom_rdsw    = {v_rdsw}',
    f'let nom_nfactor = {v_nfactor}',
    f'let nom_vsat    = {v_vsat}',
    f'let nom_eta0    = {v_eta0}',
    f'let nom_delta   = {v_delta}',
    f'let run_num     = {idx}',
    'let ax_id2 = -vds#branch',
    f'wrdata {vgs_file} vdd gate ax_id2 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta'
    
    
])
    else:
        control_lines.extend([
    f'echo " MONTE CARLO RUN: {idx}"',
    # Dynamically update the specific instance parameters
    f'alterparam vth0_nom = {v_vth0}',
    f'alterparam u0_nom = {v_u0}',
    f'alterparam rdsw_nom = {v_rdsw}',
    f'alterparam nfactor_nom = {v_nfactor}',
    f'alterparam vsat_nom = {v_vsat}',
    f'alterparam eta0_nom = {v_eta0}',
    f'alterparam delta_nom = {v_delta}',
    
    #'dc Vds 0 -1.85 -0.01  Vgs 0 -1.85 -0.185',
    'dc Vds 0 -1.85 -0.01',
    f'let nom_vth0    = {v_vth0} * (v(gate)*0 + 1)',
    f'let nom_u0      = {v_u0} * (v(gate)*0 + 1)',
    f'let nom_rdsw    = {v_rdsw} * (v(gate)*0 + 1)',
    f'let nom_nfactor = {v_nfactor} * (v(gate)*0 + 1)',
    f'let nom_vsat    = {v_vsat} * (v(gate)*0 + 1)',
    f'let nom_eta0    = {v_eta0} * (v(gate)*0 + 1)',
    f'let nom_delta   = {v_delta} * (v(gate)*0 + 1)',
    f'let run_num     = {idx} * (v(gate)*0 + 1)',
    'let ax_id0 = -vds#branch',
    #'set wr_vecnames',
    'set wr_singlescale',
    f'wrdata {vds_file} vdd gate ax_id0 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
    
    #'dc Vgs 0 -1.85 -0.01  Vds -0.01 -0.011 -1',
    #f'let out_vth0    = {v_vth0} * (v(gate)*0 + 1)',
    #f'let out_u0      = {v_u0} * (v(gate)*0 + 1)',
    #f'let out_rdsw    = {v_rdsw} * (v(gate)*0 + 1)',
    #f'let out_nfactor = {v_nfactor} * (v(gate)*0 + 1)',
    #f'let out_vsat    = {v_vsat} * (v(gate)*0 + 1)',
    #f'let out_eta0    = {v_eta0} * (v(gate)*0 + 1)',
    #f'let out_delta   = {v_delta} * (v(gate)*0 + 1)',
    #f'let run_num     = {idx} * (v(gate)*0 + 1)',
    #'let ax_id1 = -vds#branch',
    #f'wrdata {vgs_single_file} vdd gate ax_id1 out_vth0 out_u0 out_rdsw out_nfactor out_vsat out_eta0 nom_delta',
    
    #'dc Vgs 0 -1.85 -0.01  Vds 0 -1.85 -0.37',
    'dc Vgs 0 -1.85 -0.01',
    f'let nom_vth0    = {v_vth0} * (v(gate)*0 + 1)',
    f'let nom_u0      = {v_u0} * (v(gate)*0 + 1)',
    f'let nom_rdsw    = {v_rdsw} * (v(gate)*0 + 1)',
    f'let nom_nfactor = {v_nfactor} * (v(gate)*0 + 1)',
    f'let nom_vsat    = {v_vsat} * (v(gate)*0 + 1)',
    f'let nom_eta0    = {v_eta0} * (v(gate)*0 + 1)',
    f'let nom_delta   = {v_delta} * (v(gate)*0 + 1)',
    f'let run_num     = {idx} * (v(gate)*0 + 1)',
    'let ax_id2 = -vds#branch',
    f'wrdata {vgs_file} vdd gate ax_id2 nom_vth0 nom_u0 nom_rdsw nom_nfactor nom_vsat nom_eta0 nom_delta',
    
    'reset'
])
control_lines.extend(["quit", ".endc", ".end"])

with open("mc_run.spice", "w+") as f:
    # Pre-read the base template so you don't read it from disk over and over
    base_template = open(SPICE_FILE).read()
    print ('opened')


    f.seek(0)    
    f.truncate()  
    f.write(base_template)
    f.write("\n\n")
    f.write("\n".join(control_lines))
      
    f.flush()     


logFile = os.path.join(run_dir, "mc_run.log")
with open(logFile, 'w') as log:
    subprocess.run(["ngspice", "-b","-q", "mc_run.spice"], stdout=log, stderr=subprocess.STDOUT)#, capture_output=True, text=True)


#run_dir = '/home/oliviag/ngspice-skywater-sims/montecarlo/mc_output_lhc/nmos_FET_len_0p15_wid_1p6/10000_runs_data_2026-07-13_21-19-30'
#makeCSV(run_dir, N_RUNS)
#getIvVg(N_RUNS, run_dir)
#getIvVd(N_RUNS, run_dir)


end_time = time.perf_counter()
duration = end_time - start_time
print(f"Task completed in {duration:.4f} seconds.")'''
    

    #getIvVd(N_RUNS, output_dir, folder, paramV)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

