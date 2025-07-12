import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polcurvefit import polcurvefit

# === Configuration ===
BASE_FOLDER = "Tafel_Analysis"  # Folder with all subfolders
AREA_CM2 = 0.503                # 8 mm diameter electrode
EQUIV_WEIGHT = 27.0             # For Aluminum
DENSITY = 2.7                   # g/cm³
PLOT_FOLDER = "Fitted_Plots"
CSV_OUTPUT = "Tafel_Fitting_Results.csv"

os.makedirs(PLOT_FOLDER, exist_ok=True)

# === Corrosion rate calculator ===
def corrosion_rate(Icorr):
    return (0.00327 * Icorr * EQUIV_WEIGHT) / (DENSITY * AREA_CM2)

# === Search and process files ===
results = []
for root, dirs, files in os.walk(BASE_FOLDER):
    for file in files:
        if file.endswith(".xlsx"):
            filepath = os.path.join(root, file)
            relname = os.path.relpath(filepath, BASE_FOLDER)
            try:
                # Read and clean
                df = pd.read_excel(filepath, skiprows=6, names=["E", "I"])
                df = df.apply(pd.to_numeric, errors='coerce').dropna()
                df = df[np.isfinite(df["E"]) & np.isfinite(df["I"])]
                E, I = df["E"].to_numpy(), df["I"].to_numpy()

                if len(E) < 10:
                    raise ValueError("Too few valid data points after cleaning.")

                # Fitting
                Pol = polcurvefit(E, I, R=0, sample_surface=AREA_CM2 / 1e4)
                result = Pol.mixed_pol_fit(
                    window=[E.min(), E.max()],
                    apply_weight_distribution=True,
                    w_ac=0.07,
                    W=80
                )

                # Parameters
                Ecorr = result.get("Ecorr", np.nan)
                Icorr = result.get("Icorr", np.nan)
                beta_a = result.get("beta_a", np.nan)
                beta_c = result.get("beta_c", np.nan)
                Ilim = result.get("Ilim", np.nan)
                rate = corrosion_rate(Icorr)

                results.append({
                    "File": relname,
                    "Ecorr (V)": Ecorr,
                    "Icorr (A)": Icorr,
                    "Beta_a (V/dec)": beta_a,
                    "Beta_c (V/dec)": beta_c,
                    "Ilim (A)": Ilim,
                    "Corrosion Rate (mm/year)": rate
                })

                # Plotting
                fig = plt.figure(figsize=(7, 5))
                Pol.plotting(figure=fig)
                plot_name = relname.replace(os.sep, "_").replace(".xlsx", ".png")
                fig.savefig(os.path.join(PLOT_FOLDER, plot_name))
                plt.close(fig)

            except Exception as e:
                results.append({
                    "File": relname,
                    "Error": str(e)
                })

# === Save results ===
df_results = pd.DataFrame(results)
df_results.to_csv(CSV_OUTPUT, index=False)

print("✅ All done!")
print(f"📊 Results saved to: {CSV_OUTPUT}")
print(f"🖼️ Plots saved in: {PLOT_FOLDER}")
