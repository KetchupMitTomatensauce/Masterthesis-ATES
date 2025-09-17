import os
import numpy as np
import pandas as pd
from glob import glob
from pyproj import Transformer
from scipy.spatial import cKDTree
import userinput
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "./GFZ Geological Data Berlin"
EPSG_SRC = "epsg:31468"  # Gauß-Krüger Zone 4 (DHDN)
EPSG_DST = "epsg:4326"   # WGS84

def read_grid(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True, header=None, names=["x", "y", "value"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def get_layer_distribution_at_point(latitude, longitude):
    # Coordinate transformers
    to_latlon = Transformer.from_crs(EPSG_SRC, EPSG_DST, always_xy=True)
    to_xy = Transformer.from_crs(EPSG_DST, EPSG_SRC, always_xy=True)

    # Load data
    elev_files = sorted(glob(os.path.join(DATA_DIR, '*_elevation.dat')), key=lambda x: int(os.path.basename(x).split('_')[0]))
    thick_files = sorted(glob(os.path.join(DATA_DIR, '*_thickness.dat')), key=lambda x: int(os.path.basename(x).split('_')[0]))

    layers = []
    for elev_file, thick_file in zip(elev_files[:-1], thick_files):
        name = '_'.join(os.path.basename(elev_file).split('_')[1:-1])
        elev_df = read_grid(elev_file)
        thick_df = read_grid(thick_file)
        merged = elev_df.copy()
        merged['thickness'] = thick_df['value']
        merged['base'] = merged['value'] - merged['thickness']
        layers.append((name, merged))

    # LAB (no thickness)
    lab_file = elev_files[-1]
    name = '_'.join(os.path.basename(lab_file).split('_')[1:-1])
    lab_df = read_grid(lab_file)
    lab_df['thickness'] = np.nan
    lab_df['base'] = np.nan
    layers.append((name, lab_df))

    # Build KDTree from first layer
    coords = layers[0][1][['x', 'y']].values
    lons, lats = to_latlon.transform(coords[:, 0], coords[:, 1])
    geo_coords = np.column_stack([lons, lats])
    tree = cKDTree(geo_coords)

    # Query nearest point
    dist, idx = tree.query([longitude, latitude])

    # Extract values from each layer at the matched index
    result = []
    for name, df in layers:
        point = df.iloc[idx]
        result.append({
            'Layer': name,
            'Top Elevation (m)': point['value'],
            'Thickness (m)': point['thickness'],
            'Base Elevation (m)': point['base']
        })
    result = pd.DataFrame(result)
    return result

def add_geodata_to_df(df):
    geodata = pd.DataFrame({
    "Geological_unit": [
        "Holocene to Weichselian",
        "Eemian to Saalian",
        "Holstein",
        "Elsterian",
        "Miocene",
        "Cottbus",
        "Rupelian",
        "Pre-Rupelian",
        "Upper Cretaceous",
        "Lower Cretaceous",
        "Jurassic",
        "Keuper",
        "Upper Muschelkalk",
        "Middle Muschelkalk",
        "Lower Muschelkalk",
        "Upper Buntsandstein",
        "Middle Buntsandstein",
        "Lower Buntsandstein",
        "Zechstein",
        "Sedimentary Rotliegend",
        "Permocarboniferous",
        "Basement"
    ],
    "th. conductivity (W/(m·K))": [
        2.71, 2.59, 2.17, 2.35, 2.47, 2.62, 1.64, 2.48,
        2.82, 2.36, 2.71, 2.35, 2.3, 2.3, 2.3, 3.0, 2.0, 1.84,
        4.5, 3.0, 2.5, 2.2
    ],
    "vol. heatcap. of solid(MJ/(m³·K))": [
        1.57, 1.58, 1.67, 1.61, 1.56, 1.7, 1.81, 1.7,
        2.29, 2.29, 2.25, 2.32, 2.25, 2.25, 2.25, 2.19, 2.39, 2.39,
        1.94, 2.18, 2.6, 2.3
    ],
    "porosity -": [
        0.32, 0.314, 0.296, 0.304, 0.301, 0.305, 0.237, 0.297,
        0.11, 0.11, 0.189, 0.128, 0.15, 0.036, 0.12, 0.025, 0.135, 0.049,
        0.005, 0.078, 0.032, 0.01
    ],
    "hydr. conductivity (m/s)": [
        1.42e-5, 4.40e-6, 1.91e-8, 8.98e-7, 6.88e-7, 1.15e-6, 3.23e-8, 6.56e-7,
        4.81e-7, 4.81e-7, 4.81e-7, 9.62e-9, 9.62e-9, None, 5.77e-10, 6.44e-9, 5.84e-7, 1.25e-9,
        None, 5.06e-8, 8.66e-10, None
    ]})

    name_mapping = {
    "Holocene to Weichselian": "Holocene",
    "Eemian to Saalian": "Saalian",
    "Holstein": "Holstein",
    "Elsterian": "Elsterian",
    "Miocene": "Miocene",
    "Cottbus": "Cottbus",
    "Rupelian": "Rupelian",
    "Pre-Rupelian": "Pre-Rupelian",
    "Upper Cretaceous": "Upper_Cretaceous",
    "Lower Cretaceous": "Lower_Cretaceous",
    "Jurassic": "Jurassic",
    "Keuper": "Keuper",
    "Upper Muschelkalk": "Muschelkalk",
    "Middle Muschelkalk": "Muschelkalk",
    "Lower Muschelkalk": "Muschelkalk",
    "Upper Buntsandstein": "Upper_Buntsandstein",
    "Middle Buntsandstein": "Middle_Buntsandstein",
    "Lower Buntsandstein": "Lower_Buntsandstein",
    "Zechstein": "Zechstein",
    "Sedimentary Rotliegend": "Rotliegend",
    "Permocarboniferous": "Permo-Carboniferous",
    "Basement": "Pre-Permian"
    }
    geodata["Layer"] = geodata["Geological_unit"].map(name_mapping)
    df = pd.merge(df, geodata.drop(columns="Geological_unit"), on="Layer", how="left")
    return df

def ATES_ground_analysis(RUN_GEO):
    import pickle
    import os

    geo_cache_file = "groundlayers_cache.pkl"

    if RUN_GEO or not os.path.exists(geo_cache_file):
        from ATES_ground import get_layer_distribution_at_point, add_geodata_to_df
        groundlayers_heights = get_layer_distribution_at_point(userinput.latitude_input, userinput.longitude_input)
        groundlayers = add_geodata_to_df(groundlayers_heights)
        with open(geo_cache_file, "wb") as f:
            pickle.dump((groundlayers_heights, groundlayers), f)
    else:
        with open(geo_cache_file, "rb") as f:
            groundlayers_heights, groundlayers = pickle.load(f)


    filtered_groundlayers = groundlayers.copy()        
    filtered_groundlayers['hydr. conductivity (m/s)'] = groundlayers['hydr. conductivity (m/s)'].fillna(1e-10)
    # Remove duplicate rows based on "Layer" and "Top Elevation (m)", keeping one row with mean values for other columns
    cols_to_average = [col for col in filtered_groundlayers.columns if col not in ["Layer", "Top Elevation (m)"]]
    filtered_groundlayers = filtered_groundlayers.groupby(["Layer", "Top Elevation (m)"], as_index=False)[cols_to_average].mean(numeric_only=True).sort_values(by=["Top Elevation (m)"], ascending=False)
    filtered_groundlayers.reset_index(inplace=True, drop=True)

    # Calculate x-coordinates (drilling depth) based on the cumulative sum of "Thickness (m)"
    filtered_groundlayers['x_coords'] = filtered_groundlayers['Thickness (m)'].cumsum().fillna(0)

    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot horizontal lines instead of points
    import matplotlib.cm as cm

    layers = filtered_groundlayers['Layer'].unique()
    colors = cm.get_cmap('tab20', len(layers))
    layer_color_map = {layer: colors(i) for i, layer in enumerate(layers)}

    for i, row in filtered_groundlayers.iterrows():
        ax.hlines(
            y=row["hydr. conductivity (m/s)"],
            xmin=row['x_coords'] - row['Thickness (m)'],
            xmax=row['x_coords'],
            color=layer_color_map[row['Layer']],
            linewidth=5,  
            label=row['Layer'] if row['Layer'] not in ax.get_legend_handles_labels()[1] else ""
        )
    # Add legend with unique layer names
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), title="Layer Name", fontsize=10, title_fontsize=12, ncol=2)

    ax.set_yscale('log')  # Set Y-axis to logarithmic scale
    ax.set_xlabel("Depth [m]", fontsize=15)
    ax.set_ylabel("Hydraulic Conductivity (m/s)\n(logarithmic scale)", fontsize=15)
    #ax.set_title("Hydraulic Conductivity (Logarithmic Scale) of Ground Layers\n", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xlim(-20, 3100)


    # Save plot as PDF
    os.makedirs("RESULTS_PDFs", exist_ok=True)
    fig.savefig(os.path.join("RESULTS_PDFs", "hydraulic_conductivities.pdf"), format="pdf", bbox_inches="tight")

    df_ground_display = groundlayers.set_index('Layer')
    df_ground_display['Thickness (m)'] = df_ground_display['Thickness (m)'].round(1).map(lambda x: '{:g}'.format(x) if pd.notnull(x) else x)
    # Format hydraulic conductivity in exponential notation
    df_ground_display['hydr. conductivity (m/s)'] = df_ground_display['hydr. conductivity (m/s)'].map(lambda x: '{:.1e}'.format(x) if pd.notnull(x) else x)

    return (filtered_groundlayers, df_ground_display)