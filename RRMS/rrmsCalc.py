import os
import numpy as np
#from rrms_lib import *
import pandas as pd
import math
from pathlib import Path

home = Path.home()
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ashish_dir = f'{home}/ngspice-skywater-sims/outside_data/'
pMeas_Dir = f'{home}/Skywater-130nm-77K-Cryogenic-Models/cryo_data/'
hspicePath = f'{home}/ngspice-skywater-sims/outside_data'
pSim_Dir = f'{home}/ngspice-skywater-sims/pfet_mod/'
nSim_Dir = f'{home}/ngspice-skywater-sims/nfet_mod/'
simPathSmall = f'{home}/ngspice-skywater-sims/pfet_mod/l_0p35_w_1p6/vg_sweep'
original_rrms = f'{home}/ngspice-skywater-sims/RRMS/calculated_errors_for_sky130_models_dynamic_noise_removal_20260716.csv'

comp_errors = pd.read_csv(original_rrms, usecols=['relative_rmse_avg', 'relative_rmse_std'])
comp_rrms = comp_errors['relative_rmse_avg'].reset_index(drop=True)
comp_std = comp_errors['relative_rmse_std'].reset_index(drop=True)
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pmos_geometries = [
    {"display_name": "L=0.35u W=0.55u", "meas_dir": "pmos_FET_len_0p35_wid_0p55", 'sim_dir': 'l_0p35_w_0p55', 'rrms': comp_rrms[20], 'std': comp_std[20]},
    {"display_name": "L=0.35u W=1.6u",  "meas_dir": "pmos_FET_len_0.35_wid_1.6", 'sim_dir': "l_0p35_w_1p6", 'rrms': comp_rrms[19], 'std': comp_std[19]},
    {"display_name": "L=0.35u W=5u",    "meas_dir": "pmos_FET_len_0p35_wid_5", 'sim_dir': "l_0p35_w_5p0", 'rrms': comp_rrms[22], 'std': comp_std[22]}, 
    {"display_name": "L=0.5u W=0.42u",  "meas_dir": "pmos_FET_len_0p5_wid_0p42", 'sim_dir': "l_0p50_w_0p42", 'rrms': comp_rrms[21], 'std': comp_std[21]}, 
    {"display_name": "L=0.5u W=0.64u",  "meas_dir": "pmos_FET_len_0p5_wid_0p64", 'sim_dir': "l_0p50_w_0p64", 'rrms': comp_rrms[18], 'std': comp_std[18]},
    {"display_name": "L=2u W=5u",       "meas_dir": "pmos_FET_len_2_wid_5", 'sim_dir': "l_2p0_w_5p0", 'rrms': comp_rrms[14], 'std': comp_std[14]},
    {"display_name": "L=4u W=7u",       "meas_dir": "pmos_FET_len_4_wid_7", 'sim_dir': "l_4p0_w_7p0", 'rrms': comp_rrms[16], 'std': comp_std[16]},
    {"display_name": "L=8u W=0.84u",    "meas_dir": "pmos_FET_len_8_wid_0p84", 'sim_dir': "l_8p0_w_0p84", 'rrms': comp_rrms[23], 'std': comp_std[23]},
    {"display_name": "L=8u W=1.6u",     "meas_dir": "pmos_FET_len_8_wid_1p6", 'sim_dir': "l_8p0_w_1p6", 'rrms': comp_rrms[17], 'std': comp_std[17]},
    {"display_name": "L=8u W=5u",       "meas_dir": "pmos_FET_len_8_wid_5", 'sim_dir': "l_8p0_w_5p0", 'rrms': comp_rrms[13], 'std': comp_std[13]}
]
nmos_geometries = [
    {"display_name": "L=0.15u W=1.6u", "meas_dir": "nmos_FET_len_0p15_wid_1p6", 'sim_dir': "l_0p15_w_1p6", 'rrms': comp_rrms[2], 'std': comp_std[2]},
    {"display_name": "L=0.19u W=7u", "meas_dir": "nmos_FET_len_0p19_wid_7", 'sim_dir': "l_0p19_w_7p0", 'rrms': comp_rrms[5], 'std': comp_std[5]},
    {"display_name": "L=0.25u W=1.6u", "meas_dir": "nmos_FET_len_0p25_wid_1p6", 'sim_dir': "l_0p25_w_1p6", 'rrms': comp_rrms[6], 'std': comp_std[6]},
    {"display_name": "L=1u W=1.6u", "meas_dir": "nmos_FET_len_1_wid_1p6", 'sim_dir': "l_1p0_w_1p6", 'rrms': comp_rrms[1], 'std': comp_std[1]},
    {"display_name": "L=1u W=3u", "meas_dir": "nmos_FET_len_1_wid_3", 'sim_dir': "l_1p0_w_3p0", 'rrms': comp_rrms[4], 'std': comp_std[4]},
    {"display_name": "L=8u W=1.6u", "meas_dir": "nmos_FET_len_8_wid_1.6", 'sim_dir': "l_8p0_w_1p6", 'rrms': comp_rrms[8], 'std': comp_std[8]},
    {"display_name": "L=20u W=0.64u", "meas_dir": "nmos_FET_len_20_wid_0p64", 'sim_dir': "l_20p0_w_0p64", 'rrms': comp_rrms[7], 'std': comp_std[7]},
    {"display_name": "L=100u W=100u", "meas_dir": "nmos_FET_len_100_wid_100", 'sim_dir': "l_100p0_w_100p0", 'rrms': comp_rrms[3], 'std': comp_std[3]}

]


# --- FILE SUB-PAIRS ---

p_vds_pairs = [
    #('idVd_vg_0p370.txt', 'idvd_Vg-0p37.csv', '0.37'),
    #('idVd_vg_0p740.txt', 'idvd_Vg-0p74.csv', '0.74'),
    #('idVd_vg_1p110.txt', 'idvd_Vg-1p11.csv', '1.11'),
    ('idVd_vg_1p480.txt', 'idvd_Vg-1p48.csv', '1.48'),
    ('idVd_vg_1p850.txt', 'idvd_Vg-1p85.csv', '1.85')
]

p_vgs_pairs = [
    ('idVg_vd_0p010.txt', 'idvg_Vd-0p01.csv', '0.01'),
    ('idVg_vd_0p370.txt', 'idvg_Vd-0p37.csv', '0.37'),
    ('idVg_vd_0p740.txt', 'idvg_Vd-0p74.csv', '0.74'),
    ('idVg_vd_1p110.txt', 'idvg_Vd-1p11.csv', '1.11'),
    ('idVg_vd_1p480.txt', 'idvg_Vd-1p48.csv', '1.48'),
    ('idVg_vd_1p850.txt', 'idvg_Vd-1p85.csv', '1.85')
]

n_vds_pairs = [
    #('idVd_vg_0p370.txt', 'idvd_Vg0p37.csv'),
    ('idVd_vg_0p740.txt', 'idvd_Vg0p74.csv'),
    ('idVd_vg_1p110.txt', 'idvd_Vg1p11.csv'),
    ('idVd_vg_1p480.txt', 'idvd_Vg1p48.csv'),
    ('idVd_vg_1p850.txt', 'idvd_Vg1p85.csv')
]

n_vgs_pairs = [
    ('idVg_vd_0p010.txt', 'idvg_Vd0p01.csv', '0.01'),
    ('idVg_vd_0p370.txt', 'idvg_Vd0p37.csv', '0.37'),
    ('idVg_vd_0p740.txt', 'idvg_Vd0p74.csv', '0.74'),
    ('idVg_vd_1p110.txt', 'idvg_Vd1p11.csv', '1.11'),
    ('idVg_vd_1p480.txt', 'idvg_Vd1p48.csv', '1.48'),
    ('idVg_vd_1p850.txt', 'idvg_Vd1p85.csv', '1.85')
]

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def curveRRMS(I_measTot, I_simTot):
    RRMS_numSum = 0
    RRMS_denSum = 0
    N = len(I_measTot)
    
    rmse = np.sqrt(np.mean((I_simTot - I_measTot) ** 2))
    mean_meas = np.mean(np.abs(I_measTot))
    
    RRMS_k = rmse / mean_meas

    
    return RRMS_k
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def loadCurrentVals(sim_path, meas_path, vth, sim_cols, meas_cols, meas_skip):
    # Load simulation: col 5 is voltage, col 3 is Id
    sim_data = np.loadtxt(sim_path, usecols=sim_cols, unpack=True)
    v_sim, i_sim = sim_data[0], sim_data[1]

    # Load measurement: meas_cols[0] is voltage, meas_cols[1] is Id
    meas_data = np.loadtxt(
        meas_path,
        skiprows=meas_skip,
        delimiter=",",
        usecols=meas_cols,
        unpack=True,
    )
    v_meas, i_meas = meas_data[0], meas_data[1]
    
    if vth != 0:
        #i_meas = cleanCurrent(i_meas_raw)
        start_idx = np.argmax(np.abs(i_meas) > vth)
    
    
        filtered_i_sim = np.zeros(len(i_sim))
        filtered_i_sim[start_idx:] = i_sim[start_idx:]
        filtered_i_meas = np.zeros(len(i_meas))
        filtered_i_meas[start_idx:] = i_meas[start_idx:]
        
    
        min_len = min(len(filtered_i_sim), len(filtered_i_meas))
    
        return filtered_i_sim[:min_len], filtered_i_meas[:min_len]
    else:
        return i_sim, i_meas
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getCurrentData_P_Vd(sim_path, meas_path, sim_cols, meas_cols, meas_skip):
    # Load simulation: col 5 is voltage, col 3 is Id
    sim_data = np.loadtxt(sim_path, usecols=sim_cols, unpack=True)
    v_sim, i_sim = sim_data[0], sim_data[1]

    # Load measurement: meas_cols[0] is voltage, meas_cols[1] is Id
    meas_data = np.loadtxt(
        meas_path,
        skiprows=meas_skip,
        delimiter=",",
        usecols=meas_cols,
        unpack=True,
    )
    v_meas, i_meas = meas_data[0], meas_data[1]
    
    
    return i_sim, i_meas
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getCurrentData_P_Vg(sim_path, meas_path, bias_vd, device, sim_cols, meas_cols, meas_skip):

    device = (
     device.lower()
    .replace('fet_', '')
    .replace('pmos', 'hold')
    .replace('p', '.')
    .replace('hold', 'pmos')
    #.replace('_fet', '')
    )
    
    noise_csv = f'{home}/ngspice-skywater-sims/RRMS/noise_removal_log_pMOS_77K.csv'
    pmos_noise_data = pd.read_csv(noise_csv, usecols=['folder', 'file', 'noise_floor_A', 'noise_threshold_A', 'noise_cutoff_vgate_V'])
    bias_vd = float(bias_vd)
    matches = pmos_noise_data[pmos_noise_data['folder'].str.strip().str.lower() == device.strip().lower()].reset_index(drop=True)
    extracted_strings = matches['file'].str.extract(r'-drain-(-?\d+\.?\d*)')[0]
    drain_vals = pd.to_numeric(extracted_strings, errors='coerce')
    valid_mask = drain_vals.notna() & np.isclose(drain_vals.fillna(np.nan), bias_vd, atol=1e-3)

    final_match = matches[valid_mask].reset_index(drop=True)
    cutoff_vg = np.abs(final_match['noise_cutoff_vgate_V']).iloc[0]
    #print(cutoff_vg)
    if math.isnan(cutoff_vg):
        cutoff_vg = 0
        id_thrs = 0
   
    id_thrs = final_match['noise_threshold_A'].iloc[0]
    
    sim_data = np.loadtxt(sim_path, usecols=sim_cols, unpack=True)
    vg_sim, i_sim = sim_data[0], sim_data[1]

    # Load measurement: meas_cols[0] is voltage, meas_cols[1] is Id
    meas_data = np.loadtxt(
        meas_path,
        skiprows=meas_skip,
        delimiter=",",
        usecols=meas_cols,
        unpack=True,
    )
    vg_meas, i_meas = meas_data[0], meas_data[1]
    if np.abs(cutoff_vg) > 0:
        idx_m = np.argmin(np.abs(np.abs(np.array(vg_meas)) - cutoff_vg))
        idx_s = np.argmin(np.abs(np.abs(np.array(vg_sim)) - cutoff_vg))
        #print(idx_m, idx_s)
        #if (i_sim[idx_s + 1] > id_thrs) & (i_meas[idx_m + 1] > id_thrs):
        if idx_m:
            #i_sim_clean = np.zeros(len(i_sim))
            i_sim_clean = i_sim[(idx_s + 1):]
            #i_meas_clean = np.zeros(len(i_meas))
            i_meas_clean = i_meas[(idx_m + 1):]
    
            return i_sim_clean, i_meas_clean
        else:
            return i_sim, i_meas
    else:
            return i_sim, i_meas
        
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getCurrentData_N_Vg(sim_path, meas_path, bias_vd, device, sim_cols, meas_cols, meas_skip):

    device = (
     device.lower()
    .replace('fet_', '')
    #.replace('nmos', 'hold')
    .replace('p', '.')
    #.replace('hold', 'pmos')
    #.replace('_fet', '')
    )
    
    noise_csv = f'{home}/ngspice-skywater-sims/RRMS/noise_removal_log_nMOS_77K.csv'
    pmos_noise_data = pd.read_csv(noise_csv, usecols=['folder', 'file', 'noise_floor_A', 'noise_threshold_A', 'noise_cutoff_vgate_V'])
    #print(pmos_noise_data)
    bias_vd = float(bias_vd)
    matches = pmos_noise_data[pmos_noise_data['folder'].str.strip().str.lower() == device.strip().lower()].reset_index(drop=True)
    #print(matches)
    extracted_strings = matches['file'].str.extract(r'-drain(-?\d+\.?\d*)')[0]
    drain_vals = pd.to_numeric(extracted_strings, errors='coerce')
    
    valid_mask = drain_vals.notna() & np.isclose(drain_vals.fillna(np.nan), bias_vd, atol=1e-3)
    #print(valid_mask)
    final_match = matches[valid_mask].reset_index(drop=True)
    #print(final_match)

    cutoff_vg = np.abs(final_match['noise_cutoff_vgate_V']).iloc[0]
    if math.isnan(cutoff_vg):
        cutoff_vg = 0
        id_thrs = 0
    else:
        id_thrs = final_match['noise_threshold_A'].iloc[0]
        #print('exists!!')
        #print(cutoff_vg, vd)
        
    sim_data = np.loadtxt(sim_path, usecols=sim_cols, unpack=True)
    vg_sim, i_sim = sim_data[0], sim_data[1]

    # Load measurement: meas_cols[0] is voltage, meas_cols[1] is Id
    meas_data = np.loadtxt(
        meas_path,
        skiprows=meas_skip,
        delimiter=",",
        usecols=meas_cols,
        unpack=True,
    )
    vg_meas, i_meas = meas_data[0], meas_data[1]
    if np.abs(cutoff_vg) > 0:
        idx_m = np.argmin(np.abs(np.array(vg_meas) - cutoff_vg))
        idx_s = np.argmin(np.abs(np.array(vg_sim) - cutoff_vg))
    
        #if (i_sim[idx_s + 1] > id_thrs) & (i_meas[idx_m + 1] > id_thrs):
        if idx_m:
            #i_sim_clean = np.zeros(len(i_sim))
            i_sim_clean = i_sim[(idx_s + 1):]
            #i_meas_clean = np.zeros(len(i_meas))
            i_meas_clean = i_meas[(idx_m + 1):]
            #print('m idx', (idx_m+1))
            #print('s idx', (idx_s+1))
    
            return i_sim_clean, i_meas_clean
        else:
            return i_sim, i_meas
    else:
            return i_sim, i_meas
        
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#PMOS
dev_rrmsK = []
dev_sig = []
for device in pmos_geometries:
    device_simI = []
    device_measI = []
    rrmsK_curves = []
    
    p_meas = os.path.join(pMeas_Dir, device["meas_dir"])
    m_dir = device['meas_dir']
    p_simVD = os.path.join(pSim_Dir, device['sim_dir'], 'vd_sweep')
    p_simVG = os.path.join(pSim_Dir, device['sim_dir'], 'vg_sweep')
    #dev_num = device['num']
    
    #print('\n')
    #print(device["display_name"])
    
    for vdSimFile, vdMeasFile, vg in p_vds_pairs:
        col_s = (0,3)
        col_m = (0,2)
        vd_floor = 0
        
        simPathVD = os.path.join(p_simVD, vdSimFile)
        mPathVD = os.path.join(p_meas, vdMeasFile)
        
        I_simVD, I_measVD = getCurrentData_P_Vd(simPathVD, mPathVD, col_s, col_m, 1)
        
        
        vd_rrms = curveRRMS(I_measVD, I_simVD)
        rrmsK_curves.append(vd_rrms)
        #print(vd_rrms)

    for vgSimFile, vgMeasFile, vd in p_vgs_pairs:
       
        col_s = (0,3)
        col_m = (1,2)
        #floor_lib_name = f'p{dev_num}_floors'
        #floor_lib = globals().get(floor_lib_name, {})
        #vg_floor = floor_lib.get(vd, {})
        
        simPathVG = os.path.join(p_simVG, vgSimFile)
        mPathVG = os.path.join(p_meas, vgMeasFile)
        I_simVG, I_measVG = getCurrentData_P_Vg(simPathVG, mPathVG, vd, m_dir, col_s, col_m, 1)
        #if device['meas_dir'] == 'pmos_FET_len_0p35_wid_0p55':
            #print(I_simVG, I_measVG)
        
        vg_rrms = curveRRMS(I_measVG, I_simVG)
        #print(vg_rrms)
        rrmsK_curves.append(vg_rrms)
        
    dSquared = []
    #print(rrmsK_curves)
    #rrmse_now = deviceGlobalRRMSE(device_measI, device_simI)
    
    device_rrms = np.mean(rrmsK_curves)
    
    deviceRRMS = np.mean(rrmsK_curves)
    dev_rrmsK.append(deviceRRMS)
    for rrmsK in rrmsK_curves:
        underRoot = np.power((rrmsK - device_rrms), 2)
        dSquared.append(underRoot)
    sigma = np.sqrt(np.mean(dSquared))
    dev_sig.append(sigma)

print(f"\n{'Device Geometry':<25} | {'Device RRMSk':<20} | {'STD DEV (σ)':<15} | {'COMP RRMS':<15} | {'COMP STD':<15}")
for dev, rrmsk, sigma in zip(pmos_geometries, dev_rrmsK, dev_sig):
    name = dev['display_name']
    print(f"{name:<25} | {rrmsk:<20.4f} | {sigma:<15.4f} | {dev['rrms']:<15.4f} | {dev['std']:<15.4f}")
rrms_TOT = np.mean(dev_rrmsK)
sig_TOT = np.mean(dev_sig)
print(f"Total PMOS RRMS: {rrms_TOT}")
print(f"Total PMOS Std Dev: {sig_TOT}")

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#NMOS
n_dev_rrmsK = []
n_dev_sig = []
for device in nmos_geometries:
    
    device_simI = []
    device_measI = []
    #print('rrms:', device['rrms'])
    #print('std', device['std'])
    
    rrmsK_curves = []
    p_meas = os.path.join(pMeas_Dir, device["meas_dir"])
    m_dir = device['meas_dir']
    n_simVD = os.path.join(nSim_Dir, device['sim_dir'], 'vd_sweep')
    n_simVG = os.path.join(nSim_Dir, device['sim_dir'], 'vg_sweep')
    #dev_num = device['num']

    print(device["display_name"])
    for vdSimFile, vdMeasFile in n_vds_pairs:
        
        col_s = (0,3)
        col_m = (0,2)
        
        simPathVD = os.path.join(n_simVD, vdSimFile)
        mPathVD = os.path.join(p_meas, vdMeasFile)
        I_simVD, I_measVD = getCurrentData_P_Vd(simPathVD, mPathVD, col_s, col_m, 1)
           
        vd_rrms = curveRRMS(I_measVD, I_simVD)
        #print(vd_rrms)
        rrmsK_curves.append(vd_rrms)
        
          

    for vgSimFile, vgMeasFile, vd in n_vgs_pairs:
        vth = 0 #getVth(n_simVG, vgSimFile)
        col_s = (0,3)
        col_m = (1,2)

        simPathVG = os.path.join(n_simVG, vgSimFile)
        mPathVG = os.path.join(p_meas, vgMeasFile)
        I_simVG, I_measVG = getCurrentData_N_Vg(simPathVG, mPathVG, vd, m_dir, col_s, col_m, 1)
        
        
        vg_rrms = curveRRMS(I_measVG, I_simVG)
        print(vg_rrms)
        #print(vg_rrms)
        rrmsK_curves.append(vg_rrms)

    #rrmse_now = deviceGlobalRRMSE(device_measI, device_simI)
    dSquared = []
    deviceRRMS = np.mean(rrmsK_curves)
    n_dev_rrmsK.append(deviceRRMS)
    for rrmsK in rrmsK_curves:
        underRoot = np.power((rrmsK - deviceRRMS), 2)
        dSquared.append(underRoot)
    sigma = np.sqrt(np.mean(dSquared))
    n_dev_sig.append(sigma)

#print(n_dev_rrmsK)
print(f"\n{'Device Geometry':<25} | {'Device RRMSk':<20} | {'STD DEV (σ)':<15} | {'COMP RRMS':<15} | {'COMP STD':<15}")
for dev, rrmsk, sigma in zip(nmos_geometries, n_dev_rrmsK, n_dev_sig):
    name = dev['display_name']
    print(f"{name:<25} | {rrmsk:<20.4f} | {sigma:<15.4f} | {dev['rrms']:<15.4f} | {dev['std']:<15.4f}")
nmos_rrms_TOT = np.mean(n_dev_rrmsK)
nmos_sig_TOT = np.mean(n_dev_sig)
print(f"Total NMOS RRMS: {nmos_rrms_TOT}")
print(f"Total NMOS Std Dev: {nmos_sig_TOT}")



#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#combination
final_rrms = (nmos_rrms_TOT + rrms_TOT) / 2
print(f"Final Combined RRMS: {final_rrms}")
final_sig = (nmos_sig_TOT + sig_TOT) / 2
print(f"Final Combined Sigma: {final_sig}")



   