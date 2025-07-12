import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polcurvefit import polcurvefit

# === CONFIGURATION ===
BASE_FOLDER = "Tafel_Analysis"  # Your folder with subfolders
AREA_CM2 = 0.503
EQUIV_WEIGHT = 27.0
DENSITY = 2.7
PLOT_FOLDER = "Fitted_Plots"
CSV_OUTPUT = "Tafel_Fitting_Results.csv"

os.makedirs(PLOT_FOLDER, exist_ok=True)

# === CORROSION RATE ===
def corrosion_rate(Icorr):
    return (0.00327 * Icorr * EQUIV_WEIGHT) / (DENSITY * AREA_CM2)

# === CLEANING FUNCTION ===
def read_and_clean_excel(file):
    try:
        df = pd.read_excel(file, skiprows=6, usecols=[0, 1], names=["E", "I"])
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        df = df[np.isfinite(df["E"]) & np.isfinite(df["I"])]
        df = df[(df["I"] != 0)]  # Remove zero current (log10 invalid)
        return df
    except Exception as e:
        raise ValueError(f"Data read/clean error: {e}")

# === MAIN FITTING LOOP ===
results = []

for root, dirs, files in os.walk(BASE_FOLDER):
    for file in files:
        if file.endswith(".xlsx"):
            filepath = os.path.join(root, file)
            relname = os.path.relpath(filepath, BASE_FOLDER)

            try:
                df = read_and_clean_excel(filepath)
                if len(df) < 10:
                    raise ValueError("Too few valid points after cleaning.")

                E = df["E"].to_numpy()
                I = df["I"].to_numpy()

                Pol = polcurvefit(E, I, R=0, sample_surface=AREA_CM2 / 1e4)
                result = Pol.mixed_pol_fit(
                    window=[E.min(), E.max()],
                    apply_weight_distribution=True,
                    w_ac=0.07,
                    W=80
                )

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

                # Plot and save
                fig = plt.figure(figsize=(7, 5))
                Pol.plotting(figure=fig)
                plot_name = relname.replace(os.sep, "_").replace(".xlsx", ".png")
                fig.savefig(os.path.join(PLOT_FOLDER, plot_name))
                plt.close(fig)

                print(f"✅ Processed: {relname}")

            except Exception as e:
                print(f"❌ Error in file {relname}: {e}")
                results.append({
                    "File": relname,
                    "Error": str(e)
                })

# === SAVE TO CSV ===
df_results = pd.DataFrame(results)
df_results.to_csv(CSV_OUTPUT, index=False)

print("\n🎉 DONE!")
print(f"📊 Results saved in: {CSV_OUTPUT}")
print(f"🖼️ Plots saved in: {PLOT_FOLDER}")
