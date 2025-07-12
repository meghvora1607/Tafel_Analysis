import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polcurvefit import polcurvefit
import os
import io

# Constants for corrosion rate
AREA_CM2 = 0.503
EQUIV_WEIGHT = 27.0
DENSITY = 2.7

# Corrosion rate formula
def corrosion_rate(Icorr):
    return (0.00327 * Icorr * EQUIV_WEIGHT) / (DENSITY * AREA_CM2)

# App Title
st.title("Tafel Fitting App – Mixed Activation-Diffusion Control")
st.write("Upload Excel files with electrochemical data (Potential vs Current).")

# File upload
uploaded_files = st.file_uploader("Upload one or more .xlsx files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for file in uploaded_files:
        try:
            df = pd.read_excel(file, skiprows=6, names=["E", "I"]).dropna()
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
                "File": file.name,
                "Ecorr (V)": Ecorr,
                "Icorr (A)": Icorr,
                "Beta_a (V/dec)": beta_a,
                "Beta_c (V/dec)": beta_c,
                "Ilim (A)": Ilim,
                "Corrosion Rate (mm/year)": rate
            })

            # Plot
            fig = plt.figure(figsize=(7, 5))
            Pol.plotting(figure=fig)
            st.subheader(f"Fit: {file.name}")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    # Show result table
    df_results = pd.DataFrame(results)
    st.subheader("Fitting Results")
    st.dataframe(df_results)

    # CSV download
    csv = df_results.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results as CSV", csv, file_name="tafel_fit_results.csv", mime="text/csv")
