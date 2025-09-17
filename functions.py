import userinput
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from thermo import Chemical

chemical = Chemical('water', T=(userinput.t_ATES_service + 273.15))
water_density = chemical.rho  # kg/m^3
specific_heat_capacity = chemical.Cpl  # J/(kg·K)
kinematic_viscosity = chemical.nu  # m^2/s
thermal_conductivity_water = chemical.kl  # Thermal conductivity of liquid phase (W/m·K) 0.6425006607115105

def ATES_complete_setup(hydr_cond, layer_thickness, porosity, depth_ATES, temperature_depth, heat_capacity_solid, lambda_stand):
    from functions import ATES_basic_Setup, Pipe_characteristics, ATES_thermodynamic_properties

    V_dot_max_ATES, V_relation_maxmin, ATES_max_power, r_hyd, r_hyd_mid = ATES_basic_Setup(hydr_cond, layer_thickness, porosity)

    dp_max_pipes, num_wells_pipes, flow_rate_pipe, phi_pipes = Pipe_characteristics(depth_ATES, V_dot_max_ATES, kinematic_viscosity, water_density, temperature_depth, ATES_max_power)

    phi_standing, phi_charge_ATES, dp_ATES, Q_ATES_standing_mean = ATES_thermodynamic_properties(porosity, heat_capacity_solid, lambda_stand, layer_thickness, hydr_cond, r_hyd, r_hyd_mid, num_wells_pipes,  temperature_depth, V_dot_max_ATES, V_relation_maxmin, ATES_max_power, flow_rate_pipe)
 
    dp_total = dp_max_pipes + dp_ATES
    P_pump = (dp_total * V_dot_max_ATES / userinput.eta_pump) / 1000000

    print(f"power of pump: {round(P_pump,3)} MW")
    return (ATES_max_power, phi_standing, phi_charge_ATES, phi_pipes, P_pump, Q_ATES_standing_mean, num_wells_pipes)

def annuity_factor(interest=None, time=None):
    """
    Calculate the annuity of a given value over a specified period at a given interest rate.

    Parameters:
    interest (float): The annual interest rate as a percentage.
    time (float): The number of periods.

    Returns:
    float: The annuity payment.
    """
    if interest is None:
        interest = userinput.interest
    if time is None:
        time = userinput.lifetime

    if interest == 0:
        return 1 / time
    else:
        r = interest / 100
        return (r * (1 + r) ** time) / ((1 + r) ** time - 1)

def temperature_and_electricity_price():
    #time indexes
    time_index_H = pd.date_range(start=str(userinput.year_input) +"-01-01 00:00", end=str(userinput.year_input) +"-12-31  23:00", freq="H")
    #input of weatherdata
    temp_raw = xr.open_dataset("8d4f29df0089cab41cf0c19fe62be96e.nc") #input of wetherdata era5 2013
    #temperature is investigated for given location and converted to °C
    temperature = pd.Series((temp_raw.sel(latitude=userinput.latitude_input, longitude=userinput.longitude_input, method="nearest").t2m - 273.15), index = time_index_H)

    # heat temperature depending on external temperature
    temperature_heat = (userinput.t_max_input - temperature )/(userinput.t_max_input-userinput.t_min_input)*(userinput.t_heat_max - userinput.t_heat_min) + userinput.t_heat_min
    temperature_heat = pd.Series(temperature_heat.where(temperature_heat > userinput.t_heat_min, userinput.t_heat_min), index = time_index_H)

    #temperature is levelized and multiplied with maximum heat demand
    heat_demand_raw = (0.9 * (userinput.t_max_input - temperature )/(userinput.t_max_input-userinput.t_min_input) + 0.1)*userinput.heatdemand_max_input
    cooling_demand_raw = (temperature - userinput.t_ATES_service)/(userinput.t_max_input-userinput.t_min_input)*userinput.heatdemand_max_input
    #smoothing heat demand 
    heat_demand_smooth = heat_demand_raw.rolling(window=12, min_periods=1).mean()
    cooling_demand = pd.Series(cooling_demand_raw.where(cooling_demand_raw > 0, 0), index = time_index_H)
    heat_demand = pd.Series(heat_demand_smooth.where(heat_demand_smooth > 0.1*userinput.heatdemand_max_input, 0.1*userinput.heatdemand_max_input), index = time_index_H)
    
    heat_demand_sorted = heat_demand.sort_values(ascending=False)
    percentage = np.arange(1, len(heat_demand_sorted) + 1) / len(heat_demand_sorted) * 100
    # Plot 1: Sorted Heat Demand
    fig1, ax1 = plt.subplots(figsize=(15, 5))
    ax1.plot(percentage, heat_demand_sorted.values, label="Heat Demand (sorted)", linewidth=0.5)
    ax1.set_xlabel("Percentage of Entries [%]", fontsize=15)
    ax1.set_ylabel("Heat Demand [MW]", fontsize=15)
    ax1.set_ylim(bottom=0)

    #ax1.legend(loc="upper right")
    plt.savefig("RESULTS_PDFs/durationcurve.pdf", format="pdf", bbox_inches="tight")
    #plt.close(fig1)

    #spotmarket price electricity €/MWh
    electricity_spotprice_hourly_raw = pd.read_csv("csv_and_xlsx/strompreis_und_anteil_erneuerbarer_erzeugung.csv", sep=",", usecols=['Strompreis'] , decimal=".").Strompreis
    electricity_spotprice_hourly_raw.index=time_index_H

    #yearly price of electricity with taxes and duties
    electricity_yearly_price_Incl_tax = pd.read_excel("csv_and_xlsx/statistischer-bericht-energiepreisentwicklung-5619001.xlsx", sheet_name="csv-61241-16")
    electricity_yearly_price_Incl_tax_filtered = electricity_yearly_price_Incl_tax[
        (electricity_yearly_price_Incl_tax["Jahresverbrauchsklassen"] == '2 000 bis unter 20 000 MWh') &
        (electricity_yearly_price_Incl_tax["Jahr"] == userinput.year_input) &
        (electricity_yearly_price_Incl_tax["Preisarten"] == 'Durchschnittspreise ohne Umsatzsteuer u.a. abz. Steuern') 
    ].EUR_KWh.mean()*1000 #€/MWh

    electricity_duties = electricity_yearly_price_Incl_tax_filtered - electricity_spotprice_hourly_raw.mean()
    electricity_price = electricity_spotprice_hourly_raw + electricity_duties
    electricity_price_raw = electricity_spotprice_hourly_raw * electricity_price.mean()/electricity_spotprice_hourly_raw.mean()

    T_lm_sink = (temperature_heat - userinput.t_return_dh)/np.log((temperature_heat+273.15 )/(userinput.t_return_dh+273.15))
    T_lm_source = userinput.dt_hp_air/np.log((temperature+273.15 )/(temperature+273.15 - userinput.dt_hp_air))
    COP_Carnot = T_lm_sink/(T_lm_sink - T_lm_source)

    COP_hp = userinput.COP_eff * COP_Carnot

    # Plot 2: Unsorted Heat Demand, Cooling Demand, and Outside Temperature
    fig2, ax2 = plt.subplots(figsize=(15, 5))
    heat_demand.plot(ax=ax2, label="Heat Demand", linewidth=0.5)
    if userinput.LT_ATES == True:
        cooling_demand.plot(ax=ax2, label="Cooling Demand", linewidth=0.5)
    ax3 = ax2.twinx()
    temperature.plot(ax=ax3, label="Outside Temperature", color='lightgreen', linewidth=0.5)
    ax2.set_xlabel("Time", fontsize=18)
    ax2.set_ylabel("Demand [MW]", fontsize=18)
    ax2.set_ylim(bottom=-0.5)
    ax3.set_ylabel("Temperature [°C]", fontsize=18)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)
    # Combine legends from both axes into one
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=15)
    if userinput.LT_ATES:        
        plt.savefig("RESULTS_PDFs/HeatingAndCoolingDemandTemperature.pdf", format="pdf", bbox_inches="tight")
    else: 
        plt.savefig("RESULTS_PDFs/HeatingDemandTemperature.pdf", format="pdf", bbox_inches="tight")
    #plt.close(fig2)
    return (time_index_H, temperature_heat, heat_demand, cooling_demand, electricity_price, electricity_price_raw, COP_hp, temperature)

def ATES_basic_Setup(hydr_cond, layer_thickness, porosity):
    # Calculate maximum volume flow, divided by 2 to take both reservoirs into account
    r_relation_maxmin = np.sqrt(1/userinput.ATES_min_cap_fac) # ratio of maximum to minimum Volume of capacity, sqrt because of the area of a circle
    V_dot_max_ATES = hydr_cond * 2 * np.pi * layer_thickness * userinput.dp_max_ATES * 100000 / (2 * np.log(r_relation_maxmin) * water_density * 9.81)
    # Calculate corresponding maximum power per doublet
    ATES_max_power = V_dot_max_ATES * water_density * specific_heat_capacity * (userinput.t_ATES_service  - userinput.t_ATES_low) / 1000000 # MW
    if ATES_max_power > userinput.heatdemand_max_input: 
        ATES_max_power = userinput.heatdemand_max_input
        V_dot_max_ATES = ATES_max_power * 1000000 / (water_density * specific_heat_capacity * (userinput.t_ATES_service - userinput.t_ATES_low)) # m3/s

    print(f"V_dot_max_ATES: {round(V_dot_max_ATES,3)} m3/s = {round(V_dot_max_ATES*3600,3)} m3/h, ATES_max_power: {round(ATES_max_power,3)} MW")

    # Calculate water volume
    water_volume = (userinput.ATES_capacity * 1000000 * 3600) / (water_density * specific_heat_capacity * (userinput.t_ATES_service - userinput.t_ATES_low)) #m3
    r_hyd = np.sqrt(water_volume / (porosity * layer_thickness*np.pi)) #m
    r_hyd_mid = np.sqrt(water_volume*((1-userinput.ATES_min_cap_fac)/2)/(porosity*layer_thickness*np.pi)) #m
    print(f"r_hyd_max: {round(r_hyd,1)} m, r_hyd_mid: {round(r_hyd_mid,1)} m, water_volume: {round(water_volume,1)} m3")
    return (V_dot_max_ATES, r_relation_maxmin, ATES_max_power, r_hyd, r_hyd_mid)

def Pipe_characteristics(depth_ATES, V_dot_max_ATES, kinematic_viscosity, water_density, temperature_depth, ATES_max_power):
    #Determining the matching pipe characteristics and number of wells to meet maximum pressure loss

    # Calculate total length of pipes
    total_length = 2 * depth_ATES #m
    # Calculate number of wells needed for maximum pressure constraint of pipes
    num_wells_pipes = 1
    flow_rate_pipe = V_dot_max_ATES  # m^3/s

    k_value_plastic = 0.005 #https://www.schweizer-fn.de/stroemung/rauhigkeit/rauhigkeit.php #mm
    dp_max_Pa = userinput.dp_max_pipes  * 100000  # Convert bar to Pa

    Pipes_PP = pd.DataFrame({"DN (mm)": [16, 20, 25, 32, 40, 50, 63, 75, 90, 110, 125, 140, 160, 180, 200, 225, 250, 280, 315, 355, 400, 450, 500],
                    "s (mm)": [1.8, 1.9, 2.3, 2.9, 3.7, 4.6, 5.8, 6.8, 8.2, 10.0, 11.4, 12.7, 14.6, 16.4, 18.2, 20.5, 22.7, 25.4, 28.6, 32.2, 36.3, 40.9, 45.4]})#source: https://www.gfps.com/content/dam/gfps/com/specifications/de/gfps-system-specification-progef-standard-de.pdf?utm_source=chatgpt.com
    Pipes_PP["d (mm)"] = Pipes_PP["DN (mm)"] + 2 * Pipes_PP["s (mm)"]  # Calculate inner diameter in mm
    Pipes_PP["k/d"] = k_value_plastic / (Pipes_PP["DN (mm)"])  # Calculate k/d ratio

    Pipes_PP["v (m/s)"] = V_dot_max_ATES / (np.pi * (Pipes_PP["DN (mm)"] / 1000) ** 2 / 4)  # Calculate velocity in m/s
    Pipes_PP["Re"] = Pipes_PP["v (m/s)"] / (Pipes_PP["DN (mm)"] / 1000 * kinematic_viscosity)  # Calculate Reynolds number

    def calculate_lambda(row):
        if row["Re"] < 2320: #case I
            return 64 / row["Re"]
        elif row["Re"] * row["k/d"] > 1300: #case IV
            return 1 / ((2 * np.log10(3.71 * row["d (mm)"] / (row["k/d"] * row["d (mm)"]))) ** 2)
        elif 65 < row["Re"] * row["k/d"] <= 1300: #case III
            lambda_guess = 0.02  # Initial guess for lambda
            for _ in range(10):  # Iterate to refine the value
                lambda_guess = 1 / (-2 * np.log10(2.51 / (row["Re"] * lambda_guess) + row["k/d"] * 0.269))
            return lambda_guess
        elif row["Re"] * row["k/d"] <= 65: #case II
            if 2320 <= row["Re"] < 10**5: #case IIa
                return 0.3164 * row["Re"] ** -0.25
            elif 10**5 <= row["Re"] < 10**6: #case IIb
                return 0.0032 + 0.221 * row["Re"] ** -0.237
            elif row["Re"] >= 10**6: #case IIc
                lambda_guess = 0.02  # Initial guess for lambda
                for _ in range(10):  # Iterate to refine the value
                    lambda_guess = 1 / (2 * np.log10(row["Re"] * np.sqrt(lambda_guess)) - 0.8)
                return lambda_guess
            else:
                print("Error Formula 2")
        else: 
            print("Error Formula 1")

    Pipes_PP["lambda"] = Pipes_PP.apply(calculate_lambda, axis=1)
    Pipes_PP["dp"] = (water_density * Pipes_PP["v (m/s)"]**2 * Pipes_PP["lambda"] * total_length) / (2 * (Pipes_PP["d (mm)"] / 1000))  # Initialize the dp column before the loop

    while ((Pipes_PP["dp"] < dp_max_Pa).any() == False):
        num_wells_pipes += 1
        flow_rate_pipe = V_dot_max_ATES / num_wells_pipes  # m^3/s
        Pipes_PP["v (m/s)"] = flow_rate_pipe / (np.pi * (Pipes_PP["DN (mm)"] / 1000) ** 2 / 4)  # Calculate velocity in m/s
        Pipes_PP["Re"] = Pipes_PP["v (m/s)"] / (Pipes_PP["DN (mm)"] / 1000 * kinematic_viscosity)  # Calculate Reynolds number
        Pipes_PP["lambda"] = Pipes_PP.apply(calculate_lambda, axis=1)
        Pipes_PP["dp"] = (water_density * Pipes_PP["v (m/s)"]**2 * Pipes_PP["lambda"] * total_length) / (2 * (Pipes_PP["d (mm)"] / 1000))
        

    d_max_index = Pipes_PP.loc[Pipes_PP["dp"] < dp_max_Pa].index[0]
    d_ATES_ex = Pipes_PP.loc[d_max_index, "d (mm)"] # mm
    d_ATES_in = Pipes_PP.loc[d_max_index, "DN (mm)"] # mm
    dp_max_pipes = Pipes_PP.loc[d_max_index, "dp"]  # Pa

    #Determining Heat lost over Pipe surface

    Pr = (kinematic_viscosity * water_density * specific_heat_capacity) / (thermal_conductivity_water)  # Prandtl number
    Nu = 0.023 * Pipes_PP.loc[d_max_index, "Re"]**0.8 * Pr**0.4  # Nusselt number
    alpha_in = Nu * thermal_conductivity_water /(d_ATES_in*0.001) # W/m2K
    print("Pipe Requirements:")
    print(f"Number of wells per ATES : {num_wells_pipes}, volumeflow per pipe final: {round(flow_rate_pipe,4)}m3s, Pipe diameter: {round(d_ATES_in,3)} mm, Pressure drop: {round(dp_max_pipes / 100000,3)} bar, Velocity: {round(Pipes_PP.loc[d_max_index, 'v (m/s)'],3)} m/s, alpha_in: {round(alpha_in,1)} W/m2K")
    #lambda_pex = 0.4  # W/mK #https://www.warmup.co.uk/wp-content/uploads/Warmup-TS-PEX-A-v1.0-2016-09-29.pdf
    lambda_PP = 0.1 #https://www.alfa-chemistry.com/plastics/resources/plastic-thermal-conductivity-reference-table.html#:~:text=For%20plastics%2C%20this%20value%20is,polymers%20can%20achieve%20higher%20values.
    thickness_pipe = Pipes_PP.loc[Pipes_PP["d (mm)"] == (d_ATES_ex), "s (mm)"].values[0]   # Convert mm to meters
    alpha_ex = 15  # W/m2K

    U_value = 1 / ((1 / alpha_in) + (thickness_pipe / lambda_PP) + (1 / alpha_ex))  # W/m2K
    Q_loss_pipes = U_value * total_length*d_ATES_ex*0.001*np.pi*(userinput.t_ATES_service - ((temperature_depth + userinput.temperature_ground)/2))* num_wells_pipes * (1 + userinput.standing_loss_correction) #W #*1.5 due to different temperatures in reservoirs
    dt_pipe = Q_loss_pipes/num_wells_pipes * 1 / ( specific_heat_capacity * water_density * flow_rate_pipe)  # K
    # Calculate flow rate (Q = P / (ρ * c * ΔT))
    phi_pipes = ATES_max_power*1000000 / (ATES_max_power*1000000 + Q_loss_pipes) 
    print("System results:")
    print("dt: ",round(dt_pipe,3)," K, Q_loss_pipes: ", round(Q_loss_pipes,3)," W, phi: ", round(phi_pipes,6))

    return (dp_max_pipes, num_wells_pipes, flow_rate_pipe, phi_pipes)

def ATES_thermodynamic_properties(porosity, heat_capacity_solid, lambda_stand, layer_thickness, hydr_cond, r_hyd, r_hyd_mid, num_wells_pipes, temperature_depth, V_dot_max_ATES, V_relation_maxmin, ATES_max_power, flow_rate_pipe):

    #Input variation
    heat_capacity_aquifer = heat_capacity_solid * (1 - porosity) + porosity * specific_heat_capacity # J/(kg·K)

    #logarithmic factor to find mean values
    mean_fac = (1 - userinput.ATES_min_cap_fac) / 2# mean factor for capacity

    # Calculate thermal radius
    r_hyd_ATES = r_hyd/np.sqrt(num_wells_pipes) #m maximum hydraulic radius for capacity 
    r_th_ATES = r_hyd_ATES * np.sqrt(porosity * specific_heat_capacity / heat_capacity_aquifer) #m thermal radius for capacity
    r_hyd_mid_ATES = r_hyd_mid/np.sqrt(num_wells_pipes) #m minimum hydraulic radius for capacity


    #v_r_log_mean 
    v_Q = np.sqrt(porosity * np.pi * layer_thickness * water_density * specific_heat_capacity * (userinput.t_ATES_service - userinput.t_ATES_low))/ \
        (np.sqrt(userinput.ATES_capacity * mean_fac * 1000000 * 3600)) * (V_dot_max_ATES / (2 * np.pi * layer_thickness)) # m/s
    print("ATES geometry:")
    print(f"Radii: r_hyd: {round(r_hyd_ATES,1)} m, r_th: {round(r_th_ATES,1)} m, r_hyd_mid: {round(r_hyd_mid_ATES,1)} m, water_volume per storage: {round(np.pi * r_hyd_ATES**2 * layer_thickness,1)} m3, surface medium velocity: {round(v_Q,8)} m/s")

    # calculation of Q_loss_ATES
    dispersivity = 0.2 * (r_hyd_mid_ATES) ** 0.44 #m
    lambda_charge = dispersivity * v_Q * water_density * specific_heat_capacity
    heat_term = num_wells_pipes * 2 * np.pi * layer_thickness * (userinput.t_ATES_service - temperature_depth) / (np.log((1 / np.sqrt(porosity * specific_heat_capacity / heat_capacity_aquifer)))) * (1 + userinput.standing_loss_correction) # W
    Q_ATES_standing = lambda_stand * heat_term # W
    Q_ATES_standing_mean = np.mean(Q_ATES_standing*0.000001) # MW
    Q_ATES_charge = lambda_charge * heat_term # W
    
    print("Q loss ATES:")
    print(f"lambda_transmission: {round(lambda_stand,2)}, lambda_convection: {round(np.mean(lambda_charge),2)}, added faktorterm: {heat_term}")
    print(f"Q ATES standing: {round(np.mean(Q_ATES_standing*0.000001),1)} MW, Q loss charge: {round(np.mean(Q_ATES_charge*0.000001)/num_wells_pipes,1)} MW")
    # Calculate the standing loss

    phi_standing = (userinput.ATES_capacity * mean_fac * 1000000)/(Q_ATES_standing + userinput.ATES_capacity * mean_fac * 1000000)
    phi_charge_ATES = ATES_max_power/(ATES_max_power + Q_ATES_charge*0.000001)

    dp_ATES = 2 * water_density * 9.81 * np.log(V_relation_maxmin) * flow_rate_pipe / (hydr_cond * 2 * np.pi * layer_thickness) # Pressure in Pa

    print(f"phi_standing: {round(phi_standing,8)}, dp_ATES: {round(dp_ATES*0.00001,2)} bar, phi_charge_ATES: {round(phi_charge_ATES,8)}")
    return (phi_standing, phi_charge_ATES, dp_ATES, Q_ATES_standing_mean)


def calculate_ground_parameters(depth_ATES, permeability, temperature_ground = userinput.temperature_ground, heat_increase_ground=3.0):
    """
    Calculate temperature_ground, permeability, and hydr_cond for a given layer.
    """
    temperature_depth = temperature_ground + depth_ATES / 100 * heat_increase_ground  # Approximate temperature gradient of 3°C per 100m depth
    hydr_cond = permeability * 9.81 / Chemical('water', T=(temperature_depth + 273.15)).nu  # Convert Darcy to m/s
    print( "hydr_cond: ", round(hydr_cond, 8), "m/s")
    return temperature_depth, hydr_cond

def log_mean (x_max, x_min):
    """
    Calculate the logarithmic mean of two values.
    """
    if x_max == x_min:
        return x_max
    else:
        return (x_max - x_min) / np.log(x_max / x_min)