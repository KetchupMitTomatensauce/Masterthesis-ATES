import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Add this import
from thermo import Chemical
plt.ion()  # Enable interactive mode for plotting

ATES_examples = pd.DataFrame({
    "Depth": ["130", np.nan, np.nan, "240", "65", "110", np.nan, "100", np.nan, "60", "70", "75", "90", "67", "100", "1200", "60", "45", "20", "50", "260", "150"],
    "Wells": ["4,0", "7,0", "2,0", "2,0", "2,0", "10,0", "2,0", "10,0", "36,0", "2,0", "8,0", "10,0", "11,0", "2,0", "2,0", "2,0", "6,0", "18,0", "2,0", "5,0", "2,0", "2,0"],
    "Flow_m3_h": ["500,0", "1100,0", "200,0", "2,0", "100,0", "5,0", "24,0", "250,0", "3000,0", "45,0", "400,0", "120,0", "180,0", "90,0", "nan", "100,0", "272,0", "200,0", "15,0", "nan", "100,0", "20,0"],
    "Capacity_MW": ["8,3", "20,0", "1,4", "nan", "1,2", "nan", "nan", "2,8", "20,0", "0,33", "2,9", "1,3", "1,3", "0,6", "nan", "3,3", "2,0", "7,0", "nan", "nan", "2,6", "0,6"]
})

ATES_examples["Depth"] = ATES_examples["Depth"].replace("nan", np.nan, regex=False).astype(float) #m
ATES_examples["Wells"] = ATES_examples["Wells"].astype(str).str.replace(",", ".").astype(float).astype(int) #-
ATES_examples["Flow_m3_h"] = ATES_examples["Flow_m3_h"].astype(str).str.replace(",", ".").replace("nan", np.nan).astype(float)
ATES_examples["Capacity_MW"] = ATES_examples["Capacity_MW"].str.replace(",", ".").replace("nan", np.nan).astype(float)

#Constants
LT_Ates_temp = 303.15 #30° t_max
HT_Ates_temp = 363.15 #90° t_max
ground_temp = 283.15 #8°
#water = Chemical('water', T=298.15), base for constants at defined properties


kinvis_LT = Chemical('water', T=LT_Ates_temp).nu
kinvis_HT = Chemical('water', T=HT_Ates_temp).nu 
cp_LT = Chemical('water', T=LT_Ates_temp).Cpl
cp_HT = Chemical('water', T=HT_Ates_temp).Cpl
rho_LT = Chemical('water', T=LT_Ates_temp).rho
rho_HT = Chemical('water', T=HT_Ates_temp).rho
alpha_in = 600 # W/m2K
alpha_ex = 15 # W/m2K
lambda_pipe = 0.2 # W/mK
thickness_pipe = 0.005 # m


ATES_examples["Flow_per_well_m3s"] = ATES_examples["Flow_m3_h"] / ATES_examples["Wells"] / 3600

Dn_opt = np.sqrt(np.sqrt((128*kinvis_LT*rho_LT*ATES_examples["Flow_per_well_m3s"]*ATES_examples["Depth"]*2)/(np.pi*8.5)))/0.001 #mm
Dn_opt = Dn_opt.fillna(100)  # Replace NaN or invalid values with a default value (e.g., 100 mm)

ATES_examples["DN"] = np.ceil(Dn_opt / 100) * 100 #mm 
ATES_examples["dT"] = np.where(
    (ATES_examples["Flow_m3_h"] > 0) & (~ATES_examples["Flow_m3_h"].isna()),
    ATES_examples["Capacity_MW"] * 10e6 * 3600 / ATES_examples["Flow_m3_h"] / cp_LT * 1000,
    np.nan)  # K
ATES_examples["dP_real"] = (128*kinvis_LT*rho_LT*ATES_examples["Flow_per_well_m3s"]*ATES_examples["Depth"]*2)/(np.pi*(ATES_examples["DN"]*0.001)**4) # Pa, kg/m/s2
ATES_examples["dP_real"] = np.where(
    (ATES_examples["Flow_m3_h"] > 0) & (~ATES_examples["Flow_m3_h"].isna()),
    ATES_examples["dP_real"],
    np.nan)  # Pa, kg/m/s2
ATES_examples["dT"] = ATES_examples["Capacity_MW"]*3600/ATES_examples["Flow_m3_h"]/cp_LT*1000 #K
ATES_examples["U_value"] = 1/((1/alpha_in)+(thickness_pipe/lambda_pipe)+1/alpha_ex) # W/m2K  
ATES_examples["Q_pipe"] = ATES_examples["U_value"]*np.pi*ATES_examples["DN"]*ATES_examples["Depth"]*2*(LT_Ates_temp-ground_temp)*0.001 # kW
ATES_examples["phi"] = ATES_examples["Capacity_MW"]/(ATES_examples["Capacity_MW"]+ATES_examples["Q_pipe"]*0.001) # -

ATES_examples["P_Pump"] = ATES_examples["Flow_m3_h"]/3600*ATES_examples["dP_real"]# kW
