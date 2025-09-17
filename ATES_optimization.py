import os
import userinput
import numpy as np
import linopy
import pandas as pd
import matplotlib.pyplot as plt

def optimization(depth_ATES, ATES_max_power, time_index_H, temperature_heat, phi_standing, phi_charge, heat_demand, P_pump, COP_hp, electricity_price, drilling_costs=userinput.drilling_costs, Capex_TTES=userinput.Capex_TTES, interest=userinput.interest, capacity_ATES_max=None, power_ATES_min=0, power_TTES_max = None, create_LATEX = False):

    from functions import annuity_factor

    CAPEX_ATES = annuity_factor(interest, userinput.lifetime_ATES) * drilling_costs * depth_ATES * (1 + 0.01 * userinput.fom_ATES)
    OPEX_ATES = (P_pump * electricity_price/ ATES_max_power).mean()

    # OPEX costs
    opex_heatpump = electricity_price  / COP_hp
    opex_eboiler = electricity_price 

    #Capex costs generators
    from functions import annuity_factor
    capex_heatpump = (annuity_factor(interest, userinput.lifetime_hp) + 0.01 * userinput.fom_hp) * userinput.Capex_hp_decentral_input
    capex_eboiler = (annuity_factor(interest, userinput.lifetime_eb) + 0.01 * userinput.fom_eb) * userinput.Capex_eb_input



    #Adjustments for LT-ATES
    T_lm_sink_ATES = (userinput.dt_heatex)/np.log((temperature_heat+273.15 + userinput.dt_heatex)/(temperature_heat + 273.15 ))
    T_lm_source_ATES = (userinput.t_ATES_service - userinput.t_ATES_low)/np.log((userinput.t_ATES_service+273.15 )/(userinput.t_ATES_low+273.15))

    COP_Carnot_ATES = T_lm_sink_ATES/(T_lm_sink_ATES - T_lm_source_ATES)
    COP_hp_ATES = userinput.COP_eff * COP_Carnot_ATES

    opex_heatpump_ATES = electricity_price  / COP_hp_ATES
    capex_heatpump_ATES = annuity_factor(interest, userinput.lifetime_hp) * userinput.Capex_hp_ATES * (1 + 0.01 * userinput.fom_hp)





    m = linopy.Model()

    #Power
    if capacity_ATES_max == 0:
        ATES_power = m.add_variables(
            lower=0, upper=0, name="ATES_power")
        capacity_ATES = m.add_variables(
            lower=0, upper=0, name="ATES_capacity")

    elif capacity_ATES_max == None:
        capacity_ATES = m.add_variables(
            lower=0, name="ATES_capacity")
        ATES_power = m.add_variables(
            lower=power_ATES_min, name="ATES_power")
    else:
        capacity_ATES = m.add_variables(
            lower=0, upper=capacity_ATES_max, name="ATES_capacity")
        ATES_power = m.add_variables(
            lower=power_ATES_min, name="ATES_power")

#    ATES_power = m.add_variables(
#        lower=0, name="ATES_power")

    ATES_discharge = m.add_variables(
        lower=0, coords=[time_index_H], name="ATES_discharge")
    m.add_constraints(ATES_discharge - ATES_power <= 0, name="ATES_discharge_limit")

    ATES_charge = m.add_variables(
        lower=0, coords=[time_index_H], name="ATES_charge")
    m.add_constraints(ATES_charge - ATES_power <= 0, name="ATES_charge_limit")

    #Capacity and SOC
    # if capacity_ATES_max is None:
    #     capacity_ATES = m.add_variables(
    #         lower=capacity_ATES_min, name="ATES_capacity")
    # else:  
    #     capacity_ATES = m.add_variables(
    #         lower=capacity_ATES_min, upper=capacity_ATES_max, name="ATES_capacity")


    ATES_soc = m.add_variables(
        lower=0, coords=[time_index_H], name="ATES_soc") #userinput.ATES_min_cap_fac * userinput.ATES_capacity
    m.add_constraints(ATES_soc - (1-userinput.ATES_min_cap_fac) * capacity_ATES <= 0, name="ATES_cap_limit"); #capfac to account for minimum charge


    #eb
    capacity_eb = m.add_variables(name='capacity_eb', lower=0)
    gen_eb = m.add_variables(lower=0, coords=[time_index_H], name="eboiler")#upper=userinput.max_capacity_eb,
    m.add_constraints(gen_eb - capacity_eb <= 0, name="eb_limit")

    # heatpump
    capacity_hp = m.add_variables(name='capacity_hp', lower=0)
    heatpump = m.add_variables(lower=0, coords=[time_index_H], name="heatpump")#upper=userinput.max_capacity_hp,
    m.add_constraints(heatpump - capacity_hp <= 0, name="hp_limit")

    tolerance = userinput.constraint_tolerance
    dt_heat_ = temperature_heat - (userinput.t_ATES_service - userinput.dt_heatex)
    dt_heat_ATES = userinput.t_ATES_service - userinput.t_ATES_low
    dt_heat_total = dt_heat_ + dt_heat_ATES

    if userinput.LT_ATES:
        capacity_eb_ATES = m.add_variables(lower=0, upper=0, name='capacity_eb_ATES')
        gen_eb_ATES = m.add_variables(lower=0, coords=[time_index_H], name="eboiler_ATES")
        m.add_constraints(gen_eb_ATES - capacity_eb_ATES <= 0, name="eb_ATES_limit")

        #heatpump ATES
        capacity_hp_ATES = m.add_variables(lower=0,name='capacity_hp_ATES')
        heatpump_ATES = m.add_variables(lower=0, coords=[time_index_H], name="heatpump_ATES")
        m.add_constraints(heatpump_ATES - capacity_hp_ATES <= 0, name="hp_ATES_limit")

        #partition of ATES power and ATES heatpump
        m.add_constraints(heatpump_ATES * dt_heat_ATES / dt_heat_total - ATES_discharge * dt_heat_ / dt_heat_total <= tolerance, name="ATES-HP-partition1")
        m.add_constraints(ATES_discharge * dt_heat_ / dt_heat_total - heatpump_ATES * dt_heat_ATES / dt_heat_total <= tolerance, name="ATES-HP-partition2")
    else: 
        capacity_hp_ATES = m.add_variables(lower=0, upper=0, name='capacity_hp_ATES')
        heatpump_ATES = m.add_variables(lower=0, coords=[time_index_H], name='heatpump_ATES')
        m.add_constraints(heatpump_ATES - capacity_hp_ATES <= 0, name="hp_ATES_limit")

        #eboiler ATES
        capacity_eb_ATES = m.add_variables(lower=0, name='capacity_eb_ATES')
        gen_eb_ATES = m.add_variables(lower=0, coords=[time_index_H], name="eboiler_ATES")
        m.add_constraints(gen_eb_ATES - capacity_eb_ATES <= 0, name="eb_ATES_limit")
        
        #partition of ATES power and ATES eboiler
        m.add_constraints(gen_eb_ATES * dt_heat_ATES / dt_heat_total - ATES_discharge * dt_heat_ / dt_heat_total <= tolerance, name="ATES-EB-partition1")
        m.add_constraints(ATES_discharge * dt_heat_ / dt_heat_total - gen_eb_ATES * dt_heat_ATES / dt_heat_total <= tolerance, name="ATES-EB-partition2")


    #Further Constraints

    #ATES_soc_yearly
    m.add_constraints(ATES_soc.loc[time_index_H[0]] - ATES_soc.loc[time_index_H[-1]] <= 0, name="ATES_soc_yearly")

    #ATES soc
    m.add_constraints(
        #case dependant standing losses
        ATES_soc.loc[time_index_H[1:]] - (phi_standing) * ATES_soc.loc[time_index_H[:-1]] - (phi_charge) * ATES_charge.loc[time_index_H[1:]] + (1 / (phi_charge)) * ATES_discharge.loc[time_index_H[1:]] == 0, name="ATES_soc_consistency")
        #case constant heat lost in standing
        #ATES_soc.loc[time_index_H[1:]] - (phi_charge) * ATES_charge.loc[time_index_H[1:]] + (1 / (phi_charge )) * ATES_discharge.loc[time_index_H[1:]] ==  (1 - phi_standing) * userinput.ATES_capacity * (1 - userinput.ATES_min_cap_fac) / 2, name="ATES_soc_consistency")

    #costs
    capex_TTES = annuity_factor(interest, userinput.lifetime) * Capex_TTES * (1 + 0.01 * userinput.fom_TTES)

    #Capacities
    capacity_TTES = m.add_variables(
        lower=0, name="TTES_capacity")#, upper=userinput.TTES_capacity
    if power_TTES_max is None:
        TTES_charge = m.add_variables(
            lower=-userinput.TTES_power, upper=userinput.TTES_power, coords=[time_index_H], name="TTES_charge")
    else:
        TTES_charge = m.add_variables(
            lower=-power_TTES_max, upper=power_TTES_max, coords=[time_index_H], name="TTES_charge")
    TTES_soc = m.add_variables(
        lower=0, coords=[time_index_H], name="TTES_soc")#upper=capacity_TTES, 

    #Constraints
    #TTES_cap_limit
    m.add_constraints(TTES_soc - capacity_TTES <= 0, name="TTES_cap_limit")
    #TTES_yearly
    m.add_constraints(TTES_soc.loc[time_index_H[0]] - TTES_soc.loc[time_index_H[-1]] <= 0, name="TTES_soc_yearly")
    #TTES_consistency
    m.add_constraints(
        TTES_soc.sel(dim_0=time_index_H[1:]) - (1-userinput.TTES_standing_loss) * TTES_soc.shift(dim_0=1).sel(dim_0=time_index_H[1:]) -  TTES_charge.sel(dim_0=time_index_H[1:]) == 0,
        name="TTES_soc_consistency",
    )

    m.add_constraints(gen_eb + heatpump + heatpump_ATES + gen_eb_ATES + ATES_discharge - ATES_charge - TTES_charge == heat_demand.values, name="energy_balance")

    #ramping

    # m.add_constraints(
    #     gen_hp.sel(dim_0=time_index_H[1:]) - gen_hp.shift(dim_0=1).sel(dim_0=time_index_H[1:]) - userinput.ramping_hp * 0.01 * capacity_hp <= 0,
    #     name="hp_ramping_up")
    # m.add_constraints(
    #     gen_hp.sel(dim_0=time_index_H[1:]) - gen_hp.shift(dim_0=1).sel(dim_0=time_index_H[1:]) + userinput.ramping_hp * 0.01 * capacity_hp >= 0,
    #     name="hp_ramping_down")

    m.add_constraints(
        ATES_discharge.sel(dim_0=time_index_H[1:]) - ATES_discharge.shift(dim_0=1).sel(dim_0=time_index_H[1:]) <= userinput.ramping_ATES * 0.01 * ATES_max_power,
        name="ATES_discharge_ramping_up",)
    m.add_constraints(
        ATES_discharge.sel(dim_0=time_index_H[1:]) - ATES_discharge.shift(dim_0=1).sel(dim_0=time_index_H[1:]) >= -userinput.ramping_ATES * 0.01 * ATES_max_power,
        name="ATES_discharge_ramping_down",)
    # m.add_constraints(
    #     ATES_charge.sel(dim_0=time_index_H[1:]) - ATES_charge.shift(dim_0=1).sel(dim_0=time_index_H[1:]) <= userinput.ramping_ATES * 0.01 * userinput.ATES_power,
    #     name="ATES_charge_ramping_up",)
    # m.add_constraints(
    #     ATES_charge.sel(dim_0=time_index_H[1:]) - ATES_charge.shift(dim_0=1).sel(dim_0=time_index_H[1:]) >= -userinput.ramping_ATES * 0.01 * userinput.ATES_power,
    #     name="ATES_charge_ramping_down",);

    #LT-ATES case with heatpump support
    # if userinput.LT_ATES:
    #     energy_costs = (
    #         (capex_heatpump * capacity_hp)+
    #         (opex_heatpump.values * gen_hp).sum() +
    #         (capex_eboiler * capacity_eb) +
    #         (opex_eboiler.values * gen_eb).sum() +
    #         (capex_heatpump_ATES * capacity_hp_ATES) +
    #         (opex_heatpump_ATES.values * gen_hp_ATES).sum() +
    #         (capex_eboiler * capacity_eb_ATES) +
    #         (opex_eboiler.values * gen_eb_ATES).sum() +
    #         (capex_TTES * capacity_TTES) +
    #         (OPEX_ATES * (ATES_discharge + ATES_charge)).sum() +
    #         (CAPEX_ATES * ATES_power/ATES_max_power)
    #     )
    # #HT-ATES case without heatpump support
    # else:
    energy_costs = (
        (capex_heatpump * capacity_hp) +
        (opex_heatpump.values * heatpump).sum() +
        (capex_eboiler * capacity_eb) +
        (opex_eboiler.values * gen_eb).sum() +
        (capex_heatpump_ATES * capacity_hp_ATES) +
        (opex_heatpump_ATES.values * heatpump_ATES).sum() +
        (capex_eboiler * capacity_eb_ATES) +
        (opex_eboiler.values * gen_eb_ATES).sum() +
        (capex_TTES * capacity_TTES) +
        (OPEX_ATES * (ATES_discharge + ATES_charge)).sum() +
        (CAPEX_ATES * ATES_power / ATES_max_power)
    )
    m.add_objective(energy_costs)
    m.solve(solver_name="gurobi")#, OutputFlag = 0

    total_heat_supplied = (
        m.solution['heatpump'].sum() +
        m.solution['eboiler'].sum() +
        m.solution['eboiler_ATES'].sum() + 
        m.solution['heatpump_ATES'].sum()
    )
    costs_per_MWh_prod = m._objective_value/total_heat_supplied.values
    costs_per_MWh_demand = m._objective_value / heat_demand.sum()


    objective_solution = {
        "objective_value": m._objective_value,
        "total_heat_supplied": total_heat_supplied.values.item(),
        "costs_per_MWh_produced": costs_per_MWh_prod,
        "costs_per_MWh_demand": costs_per_MWh_demand,

        "capex_heatpump": m.solution.capacity_hp.item() * capex_heatpump,
        "capex_eboiler": m.solution.capacity_eb.item() * capex_eboiler,
        "capex_hp_ATES": m.solution.capacity_hp_ATES.item() * capex_heatpump_ATES,
        "capex_eb_ATES": m.solution.capacity_eb_ATES.item() * capex_eboiler,
        "capex_ATES": m.solution.ATES_power.item() / ATES_max_power * CAPEX_ATES,
        "capex_TTES": m.solution.TTES_capacity.item() * capex_TTES,

        "opex_heatpump": (m.solution.heatpump * opex_heatpump).sum().item(),
        "opex_eboiler": (m.solution.eboiler * opex_eboiler).sum().item(),
        "opex_hp_ATES": (m.solution.heatpump_ATES * opex_heatpump_ATES).sum().item(),
        "opex_eb_ATES": (m.solution.eboiler_ATES * opex_eboiler).sum().item(),
        "opex_ATES": (OPEX_ATES * (m.solution.ATES_discharge + m.solution.ATES_charge)).sum().item()
    }
    # Prepare LaTeX tables for variables, constraints, and objective
    # prepare LaTeX fragments (always define these so they exist even if create_LATEX is False)
    ates_power_bound = f"[0, ATES max power]" if ATES_max_power is not None else r"[0, $\infty$]"
    capacity_ates_bound = f"[0, {capacity_ATES_max:.3g}]" if capacity_ATES_max is not None else r"[0, $\infty$]"
    ttes_power_bound = f"[-$\infty$, $\infty$]"

    if userinput.LT_ATES:
        capacity_hp_ATES_desc = "ATES Heat Pump Energy Capacity"
        capacity_hp_ATES_value = "[0, $\infty$]"
        capacity_eb_ATES_desc = "ATES E-Boiler Capacity (0 in LT-ATES)"
        capacity_eb_ATES_value = "[0, 0]"
        partition_desc = "ATES / Heat Pump partitioning (ATES supplies heat to HP fraction)"
    else:
        capacity_hp_ATES_desc = "ATES Heat Pump Capacity (0 in HT-ATES)"
        capacity_hp_ATES_value = "[0, 0]"
        capacity_eb_ATES_desc = "ATES E-Boiler Capacity"
        capacity_eb_ATES_value = "[0, $\infty$]"
        partition_desc = "ATES / E-Boiler partitioning (ATES supplies heat to E-Boiler fraction)"

    # Variable table (display names use spaces instead of underscores)
    variables_latex = rf"""
\begin{{table}}[htbp]\label{{tab_variables}}
\centering
\small
\begin{{tabular}}{{@{{}} l @{{\hspace{{1em}}}} l @{{\hspace{{0.8em}}}} l @{{}}}}
\toprule
\textbf{{Name}} & \textbf{{Description}} & \textbf{{Bounds}} \\
\midrule
ATES power & ATES Power Capacity & {ates_power_bound} \\
ATES capacity & ATES Energy Capacity & {capacity_ates_bound} \\
ATES discharge & ATES Discharge Power (Time Series) & [0, \text{{ATES power}}] \\
ATES charge & ATES Charge Power (Time Series) & [0, \text{{ATES power}}] \\
ATES SOC & ATES State of Charge (Time Series) & $[0,\ (1-f_{{cap min}}) \cdot \text{{ATES capacity}}]$ \\
capacity eb ATES & {capacity_eb_ATES_desc} & {capacity_eb_ATES_value} \\
gen eb ATES & ATES E-Boiler Generation (Time Series) & [0, \text{{capacity eb ATES}}] \\
capacity hp ATES & {capacity_hp_ATES_desc} & {capacity_hp_ATES_value} \\
heatpump ATES & ATES Heat Pump Generation (Time Series) & [0, \text{{capacity hp ATES}}] \\
capacity eb & E-Boiler Capacity & [0, $\infty$] \\
gen eb & E-Boiler Generation (Time Series) & [0, \text{{capacity eb}}] \\
capacity hp & Heat Pump Capacity & [0, $\infty$] \\
heatpump & Heat Pump Generation (Time Series) & [0, \text{{capacity hp}}] \\
TTES capacity & TTES Energy Capacity & [0, $\infty$] \\
TTES charge & TTES Charge/Discharge (Time Series) & {ttes_power_bound} \\
TTES soc & TTES State of Charge (Time Series) & [0, \text{{TTES capacity}}] \\
\bottomrule
\end{{tabular}}
\caption{{Optimization variables of the energy system model}}
\end{{table}}
"""

    if userinput.LT_ATES:
        c_part1 = r"\(abs({{heatpump}}_{{ ATES, t}} \cdot \dfrac{\Delta T_{ATES}}{\Delta T_{total}} - {{discharge}}_{{ ATES, t}} \cdot \dfrac{\Delta T_{heat}}{\Delta T_{total}}) \leq \varepsilon \quad \forall t\)"
    else:
        c_part1 = r"\(abs({{eboiler}}_{{ATES, t}} \cdot \dfrac{\Delta T_{ATES}}{\Delta T_{total}} - {{discharge}}_{{ ATES, t}} \cdot \dfrac{\Delta T_{heat}}{\Delta T_{total}}) \leq \varepsilon \quad \forall t\)"

    constraints_latex = rf"""
\begin{{table}}[htbp]\label{{tab_constraints}}
\centering
\small
\begin{{tabular}}{{@{{}} l @{{\hspace{{1em}}}} l @{{}}}}
\toprule
\textbf{{Constraint}} & \textbf{{Equation}} \\
\midrule
Auxiliaries Partitioning & {c_part1} \\
ATES SOC Consistency & \(\ {{SOC}}_{{ATES, t}} - \phi_{{standing}}\cdot {{SOC}}_{{ATES, t-1}} \)\\ \quad &  \(\ - \phi_{{charge}}\cdot{{ATES charge}}_t +  \phi_{{charge}}^{{-1}}\cdot{{ATES discharge}}_t = 0 \quad \forall t>0\) \\
TTES SOC Consistency & \(\ {{SOC}}_{{TTES, t}} - \phi_{{standing}} \cdot {{SOC}}_{{TTES, t-1}} - {{TTES charge}}_t = 0 \quad \forall t>0\) \\
ATES SOC Yearly & \(\ {{SOC}}_{{ATES, t 0}} - {{SOC}}_{{ATES, t end}} \leq 0\) \\
TTES SOC Yearly & \(\ {{SOC}}_{{TTES, t 0}} - {{SOC}}_{{TTES, t end}} \leq 0\) \\
Energy Balance & \(\ {{heatpump}}_t + {{eboiler}}_t + {{heatpump}}_{{ATES, t}} + {{eboiler}}_{{ATES, t}} + {{discharge}}_{{ATES, t}} \)\\ \quad &  \(\ - {{charge}}_{{ATES, t}} - {{charge}}_{{TTES, t}} = {{heat demand}}_t \quad \forall t\) \\
\bottomrule
\end{{tabular}}
\caption{{Optimization Constraints of the energy system model}}
\end{{table}}
"""

    # Objective table (use spaced variable names for readability)
    objective_latex = rf"""
\begin{{table}}[htbp]\label{{tab_optimization}}
\centering
\small
\begin{{tabular}}{{@{{}} l @{{\hspace{{1em}}}} l @{{}}}}
\toprule
\textbf{{Term}} & \textbf{{Description}} \\
\midrule
minimize( & \\
$\text{{CAPEX Heat Pump}}\cdot\text{{capacity hp}}$ & Ann. CAPEX Heat Pump \\
+ $\sum \text{{OPEX Heat Pump}}\cdot\text{{heatpump}}$ & Yearly OPEX heat pump  \\
+ $\text{{CAPEX E Boiler}}\cdot\text{{capacity eb}}$ & Ann. CAPEX E Boiler \\
+ $\sum \text{{OPEX E Boiler}}\cdot\text{{gen eb}}$ & Yearly OPEX e-boiler \\
+ $\text{{CAPEX Heat Pump ATES}}\cdot\text{{capacity hp ATES}}$ & Ann. CAPEX ATES Heat Pump \\
+ $\sum \text{{OPEX Heat Pump ATES}}\cdot\text{{heatpump ATES}}$ & Yearly OPEX ATES Heat Pump \\
+ $\text{{CAPEX E Boiler}}\cdot\text{{capacity eb ATES}}$ & Ann. CAPEX ATES E Boiler \\
+ $\sum \text{{OPEX E Boiler}}\cdot\text{{gen eb ATES}}$ & Yearly OPEX ATES E Boiler \\
+ $\text{{CAPEX TTES}}\cdot\text{{TTES capacity}}$ & Ann. CAPEX TTES \\
+ $\sum \text{{OPEX ATES}}\cdot(\text{{ATES discharge}}+\text{{ATES charge}})$ & Yearly OPEX ATES Pumps \\
+ $\text{{CAPEX ATES}}\cdot\text{{ATES power}}/\text{{ATES max power}}$ & Ann. CAPEX ATES power \\
) & \\
\bottomrule
\end{{tabular}}
\caption{{Objective function elements of the energy system model (Ann = annualized)}}
\end{{table}}
"""

    # full LaTeX document fragment (concatenate fragments)
    latex_doc = variables_latex + "\n" + constraints_latex + "\n" + objective_latex

    # write file only if requested
    if create_LATEX:
        os.makedirs("RESULTS_LATEX", exist_ok=True)
        outfile = os.path.join("RESULTS_LATEX", "linopy_model_overview.tex")
        with open(outfile, "w", encoding="utf8") as f:
            f.write(latex_doc)

    return m, objective_solution

def plot_results(m, time_index_H, heat_demand, electricity_price):
    # #plot relevant graphs
    # if hasattr(m, "dual") and hasattr(m.dual, "energy_balance"):
    #     fig, axes = plt.subplots(3,1, figsize=(25,10))  
    #     m.dual.energy_balance.to_dataframe().plot(ax=axes[2], label="shadow price", marker="o", markersize=1, linestyle="None")
    #     (electricity_price).plot(ax=axes[2], label="electricity Spot market price [€/MWh]", marker="o", markersize=1, linestyle="None")
    #     axes[2].legend()  
    #     axes[2].set_title("shadow price")
    #     axes[2].axhline(0, color='black', linewidth=1, linestyle='--')
    #     axes[2].axhline(5, color='black', linewidth=0.8, linestyle=':')
    #     axes[2].axhline(10, color='black', linewidth=0.8, linestyle=':')


    # else:
    #     fig, axes = plt.subplots(1, 2, figsize=(25, 15))  
    

    # heat_demand.plot(ax=axes[0], label="Heatdemand", color="blue", marker="o", markersize=1, linestyle="None")
    # pd.Series(m.solution.eboiler, index=time_index_H).plot(ax=axes[0], label="eboiler", marker="o", markersize=1, linestyle="None")
    # pd.Series(m.solution.heatpump, index=time_index_H).plot(ax=axes[0], label="heatpump", marker="o", markersize=1, linestyle="None")
    # pd.Series(m.solution.ATES_charge, index=time_index_H).plot(ax=axes[0], label="ATES charge", marker="o", markersize=1, linestyle="None")
    # pd.Series(m.solution.TTES_charge, index=time_index_H).plot(ax=axes[0], label="TTES charge", marker="o", markersize=1, linestyle="None")
    # pd.Series(m.solution.ATES_discharge, index=time_index_H).plot(ax=axes[0], label="ATES discharge", marker="o", markersize=1, linestyle="None")
    # axes[0].set_title("Energy Components")
    # axes[0].legend()  


    # m.solution.ATES_soc.to_dataframe().plot(ax=axes[1], label="ATES soc", marker="o", markersize=1, linestyle="None")
    # m.solution.TTES_soc.to_dataframe().plot(ax=axes[1], label="TTES soc", marker="o", markersize=1, linestyle="None")
    # axes[1].set_title("storages soc ")

    fig, ax = plt.subplots(figsize=(30, 6))

    # Build positive_df depending on LT_ATES mode

    positive_df = pd.DataFrame({
        "ATES Discharge": m.solution.ATES_discharge,
        "ATES HP": m.solution.heatpump_ATES,
        "ATES E-Boiler": m.solution.eboiler_ATES,
        "E-Boiler": m.solution.eboiler,
        "TTES Discharge": -m.solution.TTES_charge.where(m.solution.TTES_charge < 0, 0),
        "Heat Pump": m.solution.heatpump.where(m.solution.heatpump > 0, 0)
    }, index=time_index_H)

    positive_df.plot.area(ax=ax, stacked=True, color=["brown", "violet", "turquoise", "black", "orange","blue"], label=["TTES Discharge", "ATES Discharge","ATES HP","ATES E-Boiler","E-Boiler", "Heat Pump"], linewidth=0.05)


    negative_df = pd.DataFrame({
        "ATES Charge": -m.solution.ATES_charge,
        "Heat Demand": -heat_demand,
        "TTES Charge": -m.solution.TTES_charge.where(m.solution.TTES_charge > 0, 0),
    }, index=time_index_H)



    negative_df.plot.area(ax=ax, stacked=True, color=["grey", "red", "green"], label=["Heat Demand", "ATES Charge", "TTES Charge"], linewidth=0.05)

    y_min = negative_df.sum(axis=1).min()
    y_max = positive_df.sum(axis=1).max()

    ax.set_ylim(y_min, y_max)
    ax.set_title("Energy Components")
    ax.legend()


    #give out relevant values
    print(f"Dimensions:\n capacity_eb: {round(m.solution.capacity_eb.item(),1)} MW, capacity_hp: {round(m.solution.capacity_hp.item(),1)} MW,"
    f"\n ATES_capacity: {round(m.solution.ATES_capacity.item(),2)} MWh, ATES_power: {round(m.solution.ATES_power.item(),2)} MW,"
    f"\n ATES_hp: {round(m.solution.capacity_hp_ATES.item(),2)} MW, ATES_eboiler: {round(m.solution.capacity_eb_ATES.item(),2)} MW,"
    f"\n capacity_TTES: {round(m.solution.TTES_capacity.item(),2)}")

    return 


