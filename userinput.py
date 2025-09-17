# location
latitude_input=52.48 #berlin
longitude_input=13.36 #berlin
#Germany = [55.1, 5.9, 47.2, 15.1]  [North, West, South, East]
#Berlin = [52.6, 13.3, 52.4, 13.5]  # [N, W, S, E]
area = [52.6, 13.3, 52.4, 13.5]  # [N, W, S, E]
#gasometer: 52.48149608511313, 13.357274468934554
#year
year_input=2013


#outside temperatures for demand
t_min_input=-14 #regulative temperature for the calculation of the heat capacity of a system
t_max_input= 15 #temperature at which heating starts
t_cooling_input= 25 #temperature at which cooling starts
#heating temperatures
t_heat_max=80 #max temperature of the heating system
t_heat_min=60 #min temperature (hot water in Summer)
#relevant for hp COP
t_return_dh = 40 #return temperature of the district heating system
dt_heatex = 2.5 #temperature difference between the heat source and the heat sink
dt_hp_air = 5 #cooling of outside air for heatpump
temperature_ground = 8 #°C


#Decision Variables to alter code execution:
LT_ATES =False
Depth_drilling_possible = False


if LT_ATES:
    t_ATES_service = 25 #°C
    t_ATES_low = 5
else:
    t_ATES_service = t_heat_min - dt_heatex  #°C
    t_ATES_low = t_return_dh - dt_heatex

#Heatpump
COP_eff = 0.5 #COP efficiency of the Carnot efficiency (heatpump)
max_capacity_hp = 20 #MW
ramping_hp = 50 #percentage of the heat production that can be ramped

#electric heating
max_capacity_eb = 50 #MW
ramping_eb = 50 #percentage of the heat production that can be ramped
electricity_tax = 60 #%

heatdemand_max_input= 7 #max heat demand in MW

#ATES
ATES_capacity = 7000 # MWh
ATES_min_cap_fac = 0.1 # MWh
#ATES_standing_loss = 0.0024  # per timestep
ATES_initial_charge = 80 # [%]
ramping_ATES = 25 #%
dp_max_pump = 6 # max pressure of pump in bar
dp_max_ATES = 5 # max pressure loss inside of ATES in bar
dp_max_pipes = 1 # max pressure loss inside of pipes in bar
standing_loss_correction = 0.0 # correction factor for standing losses
constraint_tolerance = 0.001 # MW tolerance for the constraints

perm_Darcy = 1.75 # permeability of the aquifer in Darcy
#dispersivity = 2  # m unbedingt nachprüfen

eta_pump = 0.85 # efficiency of the pump
 #€/m2MWh

#TTES
TTES_power = 50  # MW
TTES_standing_loss = 0.000077 # per timestep
TTES_initial_charge = 80 # [%]

# Economic parameters

interest = 5 #%
lifetime = 20 #years

#hp
lifetime_hp = 25 #years 
Capex_hp_decentral_input = 899488.4 #€/MWh
Capex_hp_ATES = 604065.9 #€/MWh
fom_hp = 0.2336 #%/year
vom_hp = 2.6561 #€/MWh_th
#eb
lifetime_eb = 20 #years
Capex_eb_input = 63493.3 #€/MWh
fom_eb = 1.7 #%/year
vom_eb = 1.0582 #€/MWh_th
#ATES
lifetime_ATES = 50 #years
fom_ATES = 1 #%/year
Capex_TTES_factor = 1.5 # factor for the additional TTES costs
drilling_costs = 1000 #€ per meter
#Capex_ATES_var = 100 #€/MW From lectureslides, goes down to 15€/MWh
#TTES
lifetime_TTES = 40 #years
fom_TTES = 1 #%/year
Capex_TTES =  3036.1 #€/MWh_th 


#ATES_Analysis_I
perm_analysis_factor = [1, 0.57, 0.88, 1.17, 1.71] #correlating to permeabilities 1, 1.5, 2 and 3 Darcy (based on 1.75 Darcy)
lt_analysis_factor = [1, 0.5, 0.75, 1.5, 2] #correlating to layer thicknesses between  15 and 45 m
p_analysis_factor = [1, 0.94, 0.97, 1.03, 1.06] #correlating to porosities between 0.29 and 0.32
d_analysis_factor = [1, 2, 4] #correlating to depths between ~400 and ~1600 m

#ATES_Analysis_II
elpr_range = [0.9, 1.0, 1.1] # electricity_price
drc_range = [0.5, 1.0, 1.5] #drillingcosts
CapTT_range = [1, 0.5, 2]  # Example values, adjust as needed