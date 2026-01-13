"""
This script imports data from Catman files, processes the data to remove outliers, and then generates various plots.

The script performs the following steps:
1. Set the working directory and import necessary libraries.
2. Load all Catman files from the specified folder.
3. Merge individual DataFrames into a single DataFrame.
4. Remove outliers from the data and apply smoothing.
5. Save the cleaned data in the DataFrame.
6. Generate and save various plots, including:

    - Max. displacement over time
    - All displacements over time
    - Force-time diagram
    - Force-displacement diagram
    - Deformation figure for selected time steps    

Python version used: 3.9.1 or higher
Author: Max Herbers
Date: 2025-07-11
"""


import glob
import os
import re

import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.interpolate


# Set working directory
# Path to the folder containing the data files:
data_folder = r"<put the directory path here>"
# Path to the folder where figures will be saved:
figure_folder = r"<put the directory path here>"

os.chdir(data_folder)
os.mkdir(figure_folder)

###############################################################################################################
# Function definition
###############################################################################################################

def import_catman_file(file_path, ref_tara=None):
    # Extract start time from file content (line 13, index 12)
    with open(file_path, encoding="cp1252") as f:
        lines = f.readlines()
        # Start time from line 13
        line = lines[12]
        match = re.search(r"T0\s*=\s*(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})", line) # Match format: T0 = 01.01.2020 12:00:00
        if not match:
            raise ValueError(f"Start time not found in file: {file_path}")
        dt_str = f"{match.group(3)}-{match.group(2)}-{match.group(1)} {match.group(4)}:{match.group(5)}:{match.group(6)}"
        start_time = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        # Tara values from line 28 (index 27)
        tara_line = lines[27].strip()
        tara_raw = tara_line.split("\t")
        tara_values = []
        for val in tara_raw[:17]:
            try:
                number = re.findall(r"[-+]?[0-9]*[,]?[0-9]+", val)
                tara_values.append(float(number[0].replace(",", ".")) if number else 0.0)
            except:
                tara_values.append(0.0)
    # Read data
    df = pd.read_csv(
        file_path,
        sep="\t",
        decimal=",",
        encoding="cp1252",
        skiprows=36,
        on_bad_lines="skip"
    )
    # Set column names
    df.columns = [
        "Time_1", "DMS_1", "Time_2", "Force_N", "Force_A", "IWA", "Temp_Bridge", "Temp_Ambient",
        "Time_3", "LWA_1", "LWA_2", "LWA_3", "Time_4", "LWA_4", "LWA_5", "NMA_5", "F_total", "Comment"
    ]
    # Tara correction
    tara_corr = False
    if tara_corr == True:
        # Tara correction
        for i, col in enumerate(df.columns[:17]):
            df[col] = pd.to_numeric(df[col], errors="coerce") - tara_values[i]
        # If reference tara is available (first file): Add offset back
        if ref_tara is not None:
            for i, col in enumerate(df.columns[:17]):
                df[col] += ref_tara[i]
    # Calculate time and timestamp
    df["Time_1"] = pd.to_numeric(df["Time_1"], errors="coerce")
    df["time"] = df["Time_1"].apply(lambda s: start_time + datetime.timedelta(seconds=s) if pd.notnull(s) else pd.NaT)
    return df, tara_values


def plot_deformation_figure(df_all, time_str, figure_folder):
    # Target time as datetime (only time matters)
    time_dt = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
    # Find index of the row with the minimum time difference to the desired time
    time_deltas = df_all["time"].apply(lambda t: abs((t.time().hour * 3600 + t.time().minute * 60 + t.time().second)
                                                     - (time_dt.hour * 3600 + time_dt.minute * 60 + time_dt.second)))
    idx = time_deltas.idxmin()
    # Positions of the sensors in meters
    x_sens = np.array([7.79, 15.0, 19.68, 24.3, 30.0])
    w_sens = df_all.loc[idx, ["LWA_1", "LWA_2", "LWA_3", "LWA_4", "LWA_5"]].astype(float).to_numpy()
    # Fixed point at x=0
    x_all = np.concatenate(([0.0], x_sens))
    w_all = np.concatenate(([0.0], w_sens))
    # Spline with w(0)=0 and w'(0)=0
    spline = scipy.interpolate.CubicSpline(x_all, w_all, bc_type=((1, 0.0), (2, 0.0)))
    x_plot = np.linspace(0, 30, 300)
    w_plot = -1 * spline(x_plot)
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, w_plot, label=f"Spline interpolation", color="green")
    plt.scatter(x_sens, -1 * w_sens, color="red", label="Measurement points")
    plt.xlabel("position x [m]")
    plt.ylabel("displacement w(x) [mm]")
    plt.title(f"Deformation at {df_all.loc[idx, 'time'].strftime('%H:%M:%S')} (hh:mm:ss)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, f"deformation at {time_str.replace(':', '-')}.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(figure_folder, f"deformation at {time_str.replace(':', '-')}.svg"))
    plt.show()


###############################################################################################################
# Import data
###############################################################################################################

# Load all files in the folder (e.g., all _MD_*.txt files)
all_files = sorted(glob.glob("MD_*.txt"))

# Read the first file and save reference tare
df0, ref_tara = import_catman_file(all_files[0])
all_data = [df0]

# Correct additional files with the same tare basis
for f in all_files[1:]:
    df, _ = import_catman_file(f, ref_tara=ref_tara)
    all_data.append(df)

# Merge individual DataFrames
df_all = pd.concat(all_data, ignore_index=True)

###############################################################################################################
# Data pre-processing
###############################################################################################################

# Remove outliers
lwa = df_all["LWA_4"].to_numpy()
lwa_clean = np.empty_like(lwa)
lwa_clean[0] = lwa[0]

for i in range(1, len(lwa)):
    if abs(lwa[i] - lwa[i-1]) < 1 and lwa[i] < 65:
        lwa_clean[i] = lwa[i]
    else:
        lwa_clean[i] = np.nan

# Smoothing
window = 5
lwa_clean = np.convolve(lwa_clean, np.ones(window)/window, mode='same')

# Save in DataFrame
df_all["LWA_4_clean"] = lwa_clean

###############################################################################################################
# Plots
###############################################################################################################

# Plot max displacement over time
plt.figure(figsize=(10, 6))
plt.plot(df_all["time"], df_all["LWA_4"], label="LWA_4", color="blue")
plt.plot(df_all["time"], df_all["LWA_4_clean"], label="LWA_4_clean", color="orange")
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
plt.xlabel("time [hh:mm]")
plt.ylabel("displacement at x = 24.3 m [mm]")
plt.ylim(0, 65)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "u-t (c) Max Herbers.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figure_folder, "u-t (c) Max Herbers.svg"))
plt.show()

# Plot all displacements over time
plt.figure(figsize=(10, 6))
plt.plot(df_all["time"], df_all["LWA_1"], label="LWA_1", color="blue")
plt.plot(df_all["time"], df_all["LWA_2"], label="LWA_2", color="green")
plt.plot(df_all["time"], df_all["LWA_3"], label="LWA_3", color="red")
plt.plot(df_all["time"], df_all["LWA_4_clean"], label="LWA_4_clean", color="orange")
plt.plot(df_all["time"], df_all["LWA_5"], label="LWA_5", color="purple")
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
plt.xlabel("time [hh:mm]")
plt.ylabel("displacement [mm]")
plt.ylim(-10, 65)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "u-t all (c) Max Herbers.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figure_folder, "u-t all (c) Max Herbers.svg"))
plt.show()

# Plot force-time diagram
plt.figure(figsize=(10, 6))
plt.plot(df_all["time"], df_all["F_total"], label="F_tot", color="orange")
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
plt.xlabel("time [hh:mm]")
plt.ylabel("force [kN]")
plt.ylim(0, 450)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "F-t (c) Max Herbers.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figure_folder, "F-t (c) Max Herbers.svg"))
plt.show()

# Plot force-displacement diagram
plt.figure(figsize=(10, 6))
plt.plot(df_all["LWA_4_clean"], df_all["F_total"], label="LWA_4_clean", color="orange")
plt.xlabel("displacement at x = 24.3 m [mm]")
plt.ylabel("force [kN]")
plt.xlim(0, 70)
plt.ylim(0, 450)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(figure_folder, "F-u (c) Max Herbers.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(figure_folder, "F-u (c) Max Herbers.svg"))
plt.show()

# Plot deformation figure
time = "18:00:00"
plot_deformation_figure(df_all, time, figure_folder)
