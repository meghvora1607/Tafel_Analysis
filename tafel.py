import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polcurvefit import polcurvefit
import os
import io

# Constants
AREA_CM2 = 0.503     # 8 mm diameter
EQUIV_WEIGHT = 27.0  # for aluminum
DENSITY = 2.7        # g/cm³

# Corrosion rate formula
def corrosion_rate(Icorr):
    return (0.00327 * Icorr * EQUIV_WEIGHT) / (DENSITY * AREA_CM2)

# Title
st.title("🔬 Tafel Fit App – Mixed Activation–Diffusion Control")
st.write("Upload `.xlsx` files (potential vs current), and get fitted Tafel parameters and plots.")

# File uploader
uploaded_files = st.file_uploader("Upload one or more Excel files", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    results = []

    for file in uploaded_files:
        try:
            df = pd.read_excel(file, skiprows=6, names=["E", "I"]).dropna()

            # Ensure numeric and clean
            df = df.apply(pd.to_numeric, errors='coerce').dropna()
            df = df[np.isfinite(df["E"]) & np.isfinite(df["I"])]

            E = df["E"].to_numpy()
            I = df["I"].to_numpy()

            if len(E) < 10:
                raise ValueError("Too few valid points after cleaning.")

            # Fitting
            Pol = polcurvefit(E, I, R=0, sample_surface=AREA_CM2 / 1e4)
            result = Pol.mixed_pol_fit(
                window=[E.min(), E.max()],
                apply_weight_distribution=True,
                w_ac=0.07,
                W=80
            )

            # Extract parameters
            Ecorr = result.get("Ecorr", np.nan)
            Icorr = result.get("Icorr", np.nan)
            beta_a = result.get("beta_a", np.nan)
            beta_c = result.get("beta_c", np.nan)
            Ilim = result.get("Ilim", np.nan)
            rate = corrosion_rate(Icorr)

            # Store
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
            st.subheader(f"📉 Fitted Plot – {file.name}")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error in file {file.name}: {e}")

    # Show table
    if results:
        df_results = pd.DataFrame(results)
        st.subheader("📊 Fitted Parameters")
        st.dataframe(df_results)

        # Download button
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", csv, file_name="tafel_fits.csv", mime="text/csv")
