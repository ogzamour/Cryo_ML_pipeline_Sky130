import numpy as np
import subprocess
import csv
import os
from datetime import datetime
from pathlib import Path
import time
#from getCSV import makeCSV
#from getIV import getIvVd
from scipy.stats import qmc


#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def getHypercube(n_sampples, n_var)
    # Define number of samples (N) and parameters (d)
    num_samples = 20
    num_vars = 2 # e.g., Width (W) and Length (L)
    
    # Generate Latin Hypercube samples
    sampler = qmc.LatinHypercube(d=num_vars)
    sample_matrix = sampler.random(n=num_samples)
    
    # Scale samples to actual parameter ranges
    # W: [500n, 1.5u]
    # L: [100n, 500n]
    scaled_matrix = qmc.scale(
        sample_matrix,
        [500e-9, 100e-9],
        [1.5e-6, 500e-9]
    )
    
    # Export to a text file for ngspice to read
    np.savetxt('lhs_parameters.txt', scaled_matrix, fmt='%.16e')
    return

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
N_RUNS = 10000
SPICE_FILE = "paramMC_vds_sweep.spice"
PARENT_DIR = Path("/home/oliviag/ngspice-skywater-sims/mc_vds_output")

folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = PARENT_DIR / folder
output_dir.mkdir(parents=True, exist_ok=True)

vth0 = 1
u0 = 1
rdsw = 1
nfactor = 1
vsat = 1
eta0 = 1
delta = 1

paramV = []

if vth0: paramV.append("vth0")
if u0: paramV.append("u0")
if rdsw: paramV.append("rdsw")
if nfactor: paramV.append("nfactor")
if vsat: paramV.append("vsat")
if eta0: paramV.append("eta0")
if delta: paramV.append("delta")



start_time = time.perf_counter()



cmd = [f"""
        .control
        run
        set filetype=ascii
        *showmod m.xm1.msky130_fd_pr__nfet_01v8_lvt : vth0 u0 rdsw nfactor vsat eta0 delta
    
        *running montecarlo sim, 1 param at at a time
        showmod m.xm1.msky130_fd_pr__nfet_01v8_lvt : vth0 u0 rdsw nfactor vsat eta0 delta > ogparam.txt
        set raw_array = ( `cat ogparam.txt` )
          define agauss(mean, sigma) (mean + sigma * sgauss(0))  
          set nom_vth0 = $raw_array[11]
          set nom_u0 = $raw_array[13]
          set nom_rdsw = $raw_array[15]
          set nom_nfactor = $raw_array[17]
          set nom_vsat = $raw_array[19]
          set nom_eta0 = $raw_array[21]
          set nom_delta = $raw_array[23] 
        *showmod m.xm1.msky130_fd_pr__nfet_01v8_lvt : vth0 u0 rdsw nfactor vsat eta0 delta 
        *echo "$nom_vth0"
        set outDir = {output_dir}
        *controls
          let mc_vth0 = {vth0}
          let mc_u0 = {u0}
          let mc_rdsw = {rdsw}
          let mc_nfactor = {nfactor}
          let mc_vsat = {vsat}
          let mc_eta0 = {eta0}
          let mc_delta = {delta}
        let i_max = {N_RUNS}
        let idx = 0
        let i1 = 1
        """]
cmd.append("""  
      let s_vth0 = 0.319943
      let s_u0 = 0.0430804
      let s_rdsw = 103.65
      let s_nfactor = 0.0351348
      let s_vsat = 78424.2
      let s_eta0 = -0.191127
      let s_delta = 0.01
      """)




cmd.append('dowhile (idx < i_max)')

cmd.append("""   
    *echo "========================================="
    echo " MONTE CARLO RUN: $&idx" 
    *echo "========================================="
    let v_vth0 = $nom_vth0
    let v_u0 = $nom_u0
    let v_rdsw = $nom_rdsw
    let v_nfactor = $nom_nfactor
    let v_vsat = $nom_vsat
    let v_eta0 = $nom_eta0
    let v_delta = $nom_delta
    """)

cmd.append("""
    let vth0_min = 0.9 * s_vth0
    let vth0_max = 1.1 * $nom_vth0
    let u0_min = 0.9 * s_u0
    let u0_max = 1.1 * $nom_u0
    let rdsw_min = 0.9 * s_rdsw
    let rdsw_max = 1.1 * $nom_rdsw
    let nf_min = 0.9 * s_nfactor
    let nf_max = 1.1 * $nom_nfactor
    let vsat_min = 0.9 * s_vsat
    let vsat_max = 1.1 * $nom_vsat
    let eta0_min = 0.9 * s_eta0
    let eta0_max = 1.1 * $nom_eta0
    let delta_min = 0.9 * s_delta
    let delta_max = 1.1 * $nom_delta
    print idx
           """)
cmd.append("""
    let u = agauss(0,1)
    let u = (u + 3) / 6
    let u = (u > 1)*1 + (u <= 1)*((u < 0) * 0 + (u >= 0) * u)

    if (mc_vth0)
      *let v_vth0 = agauss($nom_vth0, $s_vth0)
      let v_vth0 = {vth0_min} + ({vth0_max} - {vth0_min}) * u
      *echo $&v_vth0
    endif
    if (mc_u0)
      *let v_u0 = agauss($nom_u0, $s_u0)
      let v_u0 = {u0_min} + ({u0_max} - {u0_min}) * u
    endif
    if (mc_rdsw)
      *let v_rdsw = agauss($nom_rdsw, $s_rdsw)
      let v_rdsw = {rdsw_min} + ({rdsw_max} - {rdsw_min}) * u
    endif
    if (mc_nfactor)
      *let v_nfactor = agauss($nom_nfactor, $s_nfactor)
      let v_nfactor = {nf_min} + ({nf_max} - {nf_min}) * u
    endif
    if (mc_vsat)
      *let v_vsat = agauss($nom_vsat, $s_vsat)
      let v_vsat = {vsat_min} + ({vsat_max} - {vsat_min}) * u
    endif
    if (mc_eta0)
      *let v_eta0 = agauss($nom_eta0, $s_eta0)
      let v_eta0 = {eta0_min} + ({eta0_max} - {eta0_min}) * u
    endif
    if (mc_delta)
      *let v_delta = agauss($nom_delta, $s_delta)
      let v_delta = {delta_min} + ({delta_max} - {delta_min}) * u
    endif
    
    *showmod m.xm1.msky130_fd_pr__nfet_01v8_lvt : vth0 u0 rdsw nfactor vsat eta0 delta 
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[vth0]    = $&v_vth0
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[u0]      = $&v_u0
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[rdsw]    = $&v_rdsw
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[nfactor] = $&v_nfactor
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[vsat]    = $&v_vsat
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[eta0]    = $&v_eta0
    altermod @m.xm1.msky130_fd_pr__nfet_01v8_lvt[delta]   = $&v_delta
    
    *showmod m.xm1.msky130_fd_pr__nfet_01v8_lvt : vth0 u0 rdsw nfactor vsat eta0 delta
    dc VDS 0 1.85 0.01

    let out_vth0    = v_vth0*(v(drain)*0 + 1)
    let out_u0      = v_u0 * (v(DRAIN)*0 + 1)
    let out_rdsw    = v_rdsw * (v(DRAIN)*0 + 1)
    let out_nfactor = v_nfactor * (v(DRAIN)*0 + 1)
    let out_vsat    = v_vsat * (v(DRAIN)*0 + 1)
    let out_eta0    = v_eta0 * (v(DRAIN)*0 + 1)
    let out_delta   = v_delta * (v(DRAIN)*0 + 1)
    let run_num     = $&idx * (v(DRAIN)*0 + 1)
    


   
    set wr_vecnames
    set wr_singlescale

  
    *set raw_file = "mc_output/output{$&idx}.dat"
    set raw_file = "$outDir/output{$&idx}.dat"
    wrdata $raw_file run_num v(gate) i(vds) out_vth0 out_u0 out_rdsw out_nfactor out_vsat out_eta0
    *let hold = idx
    let idx = idx + i1
    

    
  end


quit
.endc

.end
""")

with open("mc_run.spice", "w+") as f:
    # Pre-read the base template so you don't read it from disk over and over
    base_template = open(SPICE_FILE).read()
    print ('opened')


    f.seek(0)      # Move the "cursor" back to the very top of the file
        #print('fseek')
    f.truncate()  # Delete everything currently in the file
        #print('truncate')
        # Write the fresh data
    f.write(base_template)
    f.write("\n\n")
    f.write("\n".join(cmd))
       # print('written')
    f.flush()     # Force Python to actually write the data to the hard drive immediately
       # print('flush')
        # Run ngspice safely
subprocess.run(["ngspice", "-b","-q", "mc_run.spice"])#, capture_output=True, text=True)
#end = time.time()
#print("one run time:", end - start)
      #  print('run')
end_time = time.perf_counter()
duration = end_time - start_time

print(f"Task completed in {duration:.4f} seconds.")

makeCSV(output_dir, N_RUNS)
getIvVd(N_RUNS, output_dir, folder, paramV)
