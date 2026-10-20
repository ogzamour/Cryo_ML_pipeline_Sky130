import numpy as np
import glob
import re
import os
import pandas as pd
import time

def parseCurvesVg(simPath, n_runs):
    n_runs = n_runs - 1
    parsedIvVg = f"/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6/{n_runs}_runs_IvVg_1.48_vd_data.csv"
    parsedIvVd = f"/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6/{n_runs}_runs_IvVd_1.48_vg_data.csv"
    parsedParam = f"/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6/{n_runs}_runs_param_data.csv"

    for file_path in [parsedIvVg, parsedParam]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleared previous output: {file_path}")
            except OSError as e:
                print(f"Warning: Could not delete {file_path}: {e}")
                
    for file_path in [parsedIvVg, parsedParam]:
        out_dir = os.path.dirname(file_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    
    initialPathVg = os.path.join(simPath, "vgs_sweep_0.dat")
    #vgs_col = pd.read_csv(initialPathVg, sep=r"\s+", header=None, skiprows=[0], usecols=[0], names=['vSweep'])
    vgs_col = pd.read_csv(
    initialPathVg, 
    sep=r"\s+", 
    header=None, 
    skiprows=[0], 
    usecols=[0], 
    names=['vSweep'],
    engine='python'  # This bypasses C-engine truncation issues
)
    vgs_col.name = "Vgs (V)"
    print(vgs_col)
    
    ivVg_cols = []
    vth0_col = []
    u0_col = []
    rdsw_col = []
    nf_col = []
    vsat_col = []
    eta0_col = []
    delta_col = []
    missing_runs = []
    #n_runs = 3
    for run in range(n_runs):
        run_file = os.path.join(simPath, f"vgs_sweep_{run}.dat")
        success = False
        for attempt in range(5):
            if os.path.exists(run_file):
                try:
                    #size1 = os.path.getsize(run_file)
                    #time.sleep(0.02)
                    #size2 = os.path.getsize(run_file)
                    # Read column index 3, skip first row, and use iloc[:, 0] to safely access the single loaded column
                    data = pd.read_csv(run_file, sep=r"\s+", header=None, skiprows=[0]) #usecols=[3]
                    current = data.iloc[:, 3].reset_index(drop=True)
                    current.name = f"run_{run} (A)"
                    ivVg_cols.append(current)
                    #print(current)
                    
                    vth0_nom = data.iloc[1, 4]
                    vth0_col.append(vth0_nom)
                    u0_nom = data.iloc[1, 5]
                    u0_col.append(u0_nom)
                    rdsw_nom = data.iloc[1, 6]
                    rdsw_col.append(rdsw_nom)
                    nf_nom = data.iloc[1, 7]
                    nf_col.append(nf_nom)
                    vsat_nom = data.iloc[1, 8]
                    vsat_col.append(vsat_nom)
                    eta0_nom = data.iloc[1, 9]
                    eta0_col.append(eta0_nom)
                    delta_nom = data.iloc[1, 10]
                    delta_col.append(delta_nom)
                    
                    success = True
                    break  # Exit retry loop on success
                except (OSError, pd.errors.EmptyDataError):
                # Catches cases where file exists but is empty or locked mid-write
                    time.sleep(0.05)
            #else:
                #missing_runs.append(run)
            # File doesn't exist yet, wait and retry
                #if attempt < 4:
                    #time.sleep(0.05)
                
               
                    
        if not success:
            missing_runs.append(run)
            #print(f'Run {run} exists but having issues.')
            
        if (run + 1) % 10000 == 0:
            print(f"Processed {run + 1}/{n_runs} runs...")
            
    
    if missing_runs:
        print(f"\nCompleted with warnings: {len(missing_runs)} runs were missing/unreadable.")
        print(f"First 10 skipped runs: {missing_runs[:10]}")
        trainingPath = '/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6'
        
        # Save missing runs to a text file
        missing_file = os.path.join(trainingPath, "missing_runs.txt")
        
        # Clear existing missing runs file if it exists
        if os.path.exists(missing_file):
            try:
                os.remove(missing_file)
                print(f"Cleared existing missing runs file: {missing_file}")
            except OSError as e:
                print(f"Error removing {missing_file}: {e}")
                
        # Write fresh missing runs
        with open(missing_file, "w") as f:
            for run in missing_runs:
                f.write(f"run_{run}\n")
        print(f"Saved missing runs list to: {missing_file}")
    missingNum = len(missing_runs)
    newNum = n_runs - missingNum
    for output_file in [parsedIvVg, parsedParam]:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Cleared existing output file: {output_file}")
            except OSError as e:
                print(f"Error removing {output_file}: {e}")
    if len(vgs_col) == 185: 
        print('entered parsing')
        output = pd.concat([vgs_col] + ivVg_cols, axis=1)
        print('saved concat')
        output.to_csv(parsedIvVg, index=False)
        print('saved iv curves')
    else:
        print('parsing error for iv')
    if len(vth0_col) < 2:
        print("params empy")
    else:
        print('params exist')
        param_output = pd.DataFrame({
            'vth0_nom': vth0_col,
            'vsat_nom': vsat_col,
            'rdsw_nom': rdsw_col,
            'u0_nom': u0_col,
            'nf_nom': nf_col,
            'eta0_nom': eta0_col,
            'delta_nom': delta_col
        })
        print('param data frame made')
        param_output.to_csv(parsedParam, index=False)
        print(f"Successfully saved parameters to {parsedParam}")

    #paramFile = os.path.join(simPath, 'lhs_parameters.txt')
    #params = pd.read_csv(paramFile, sep=r"\s+", header=None)
    #params.to_csv(parsedParam, index=True)

    print('done')
    #return
    return newNum

#def completeDataSet():
'''def parseCurvesVg(simPath, n_runs):
    

    parsedIvVg = f"/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6/{n_runs}_runs_IvVg_1.48_vd_data.csv"
    parsedIvVd = f"/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6/{n_runs}_runs_IvVd_1.48_vg_data.csv"
    parsedParam = f"/home/oliviag/ngspice-skywater-sims/ml_method1/training_data/nmos_FET_len_0p15_wid_1p6/{n_runs}_runs_param_data.csv"

    for file_path in [parsedIvVg, parsedParam]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Cleared previous output: {file_path}")
            except OSError as e:
                print(f"Warning: Could not delete {file_path}: {e}")
    for file_path in [parsedIvVg, parsedParam]:
        out_dir = os.path.dirname(file_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
    
    initialPathVg = os.path.join(simPath, "vgs_sweep_0.dat")
    vgs_col = pd.read_csv(initialPathVg, sep=r"\s+", header=None, skiprows=[0], usecols=[0], names=['vSweep'])
    vgs_col.name = "Vgs (V)"
    ivVg_cols = []
    missing_runs = []
    
    for run in range(n_runs):
        run_file = os.path.join(simPath, f"vgs_sweep_{run}.dat")
        for attempt in range(3):
            try:
                # Read ONLY column index 3 to optimize RAM usage across 100k files
                data = pd.read_csv(run_file, sep=r"\s+", header=None, skiprows=[0], usecols=[3])
                #current = data[3]
                current = data.iloc[:, 0].reset_index(drop=True)
                current.name = f"run_{run} (A)"      
                ivVg_cols.append(current)
                success = True
                break  # Exit retry loop on success
            except (FileNotFoundError, OSError):
                if attempt < 2:
                    time.sleep(0.05)
        if not success:
            missing_runs.append(run)
        if (run + 1) % 10000 == 0:
            print(f"Processed {run + 1}/{n_runs} runs...")
        if missing_runs:
            print(f"\nCompleted with warnings: {len(missing_runs)} runs were missing/unreadable.")
            print(f"First 10 skipped runs: {missing_runs[:10]}")
       
        #data = pd.read_csv(run_file)
        #data = pd.read_csv(run_file, sep=r"\s+", header=None)
       
        #current = data[3]
        #current.name = f"run_{run} (A)"      
        #ivVg_cols.append(current)
    for output_file in [parsedIvVg, parsedParam]:
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"Cleared existing output file: {output_file}")
            except OSError as e:
                print(f"Error removing {output_file}: {e}")
    output = pd.concat([vgs_col] + ivVg_cols, axis=1)
    output.to_csv(parsedIvVg, index=False)

    paramFile = os.path.join(simPath, 'lhs_parameters.txt')
    params = pd.read_csv(paramFile, sep=r"\s+", header=None)
    params.to_csv(parsedParam, index=True)

    print('done')

    return'''



    
        
        
        
    
    
    