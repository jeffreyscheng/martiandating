"""Parse Nakhla #2 from Swindle & Olson Table A3, impute step durations."""
import numpy as np
import pandas as pd

# ── Nakhla #2 data from Table A3 ────────────────────────────────
# Columns: Temp(°C), 40Ar(10^-8 cm3 STP), 39Ar (with uncertainty)
# 39Ar values and uncertainties read from the table
nakhla2_data = [
    (250,  194, 1.5160, 0.0041),
    (275,   20, 3.107,  0.011),
    (300,   18, 3.445,  0.012),
    (325,   25, 2.699,  0.008),
    (350,   15, 4.999,  0.018),
    (375,    9, 8.03,   0.05),
    (400,   13, 8.098,  0.036),
    (425,   19, 7.827,  0.025),
    (450,   29, 7.492,  0.020),
    (475,   42, 7.148,  0.011),
    (500,   62, 7.059,  0.014),
    (525,   82, 7.034,  0.009),
    (537,   50, 7.206,  0.013),
    (551,   54, 7.193,  0.012),
    (562,   36, 7.368,  0.014),
    (575,   41, 7.442,  0.012),
    (600,   60, 7.557,  0.012),
    (625,   60, 7.694,  0.012),
    (650,   57, 7.746,  0.011),
    (675,   59, 7.698,  0.008),
    (700,   44, 7.830,  0.010),
    (725,   35, 7.859,  0.016),
    (750,   26, 7.852,  0.025),
    (800,  272, 8.035,  0.017),
    (900,   41, 8.672,  0.013),
    (1000,  20, 10.887, 0.032),
    (1100,   9, 15.54,  0.10),
    (1200,   7, 41.92,  0.35),
]

df2 = pd.DataFrame(nakhla2_data, columns=['Temp', '40Ar', '39Ar', 'std_39Ar'])
total = df2['39Ar'].sum()
df2['dF'] = df2['39Ar'] / total
df2['F'] = df2['dF'].cumsum()
df2['T_K'] = df2['Temp'] + 273.15

def y_from_F(F):
    if F < 0 or F > 1: return np.nan
    if F < 0.85:
        a = 6 / np.sqrt(np.pi)
        return ((a - np.sqrt(max(a * a - 12 * F, 0))) / 6) ** 2
    return -(1 / np.pi**2) * np.log(max((1 - F) * np.pi**2 / 6, 1e-30))

df2['y'] = df2['F'].apply(y_from_F)

# ── Impute step durations ───────────────────────────────────────
# Method: use Nakhla #1's CORRECTED imputed durations at matching temps,
# interpolate for temps unique to Nakhla #2 (537, 551, 562, 725, 750, 800, 900, 1000+)

# First, recompute Nakhla #1 durations CORRECTLY (fix the off-by-one)
df1 = pd.read_csv('nakhla1_parsed.csv')[['Temp', '39Ar', 'std_39Ar', 'seconds_per_extraction_step']]
total1 = df1['39Ar'].sum()
df1['dF'] = df1['39Ar'] / total1
df1['F'] = df1['dF'].cumsum()
df1['y'] = df1['F'].apply(y_from_F)

# S&W Figure 1 ln(D/rho^2) values (22 values for steps up to 700°C)
lnD_fig1 = [-15.65, -16.40, -16.1, -16.4, -15.8, -15.38, -14.9, -14.3, -13.7,
            -13.3, -12.6, -11.8, -11.2, -11.0, -10.9, -10.8, -10.9, -10.8,
            -10.7, -10.3, -10.2, -10.1]

# CORRECT formula: dt_i = Δy_i / exp(lnD_i) where Δy_i = y_i - y_{i-1}
df1['lnD_true'] = lnD_fig1 + [np.nan] * (len(df1) - len(lnD_fig1))
df1['dy'] = df1['y'].diff()
df1['dt_corrected'] = df1['dy'] / np.exp(df1['lnD_true'])

print("=== Nakhla #1 duration comparison ===")
print("Temp  dt_old (off-by-1)  dt_corrected   ratio")
for _, r in df1.head(22).iterrows():
    old = r['seconds_per_extraction_step']
    new = r['dt_corrected']
    ratio = old / new if new > 0 and not np.isnan(new) else np.nan
    print(f"{r.Temp:4.0f}  {old:12.1f}       {new:12.1f}       {ratio:.2f}" if not np.isnan(ratio) else f"{r.Temp:4.0f}  {old:12.1f}       {'NaN':>12s}       NaN")

# Save corrected Nakhla #1
df1_out = df1[['Temp', '39Ar', 'std_39Ar']].copy()
df1_out['seconds_per_extraction_step'] = df1['dt_corrected']
df1_out.to_csv('nakhla1_parsed_fitted.csv', index=False)
print("\nSaved corrected nakhla1_parsed_fitted.csv")

# Build interpolation for Nakhla #2
# Use corrected durations from Nakhla #1 at matching temps, interpolate others
from scipy.interpolate import interp1d
valid1 = df1[df1['dt_corrected'].notna() & (df1['dt_corrected'] > 0)]
dt_interp = interp1d(valid1['Temp'].values, valid1['dt_corrected'].values,
                     kind='linear', fill_value='extrapolate')

# For Nakhla #2, apply durations
df2['seconds_per_extraction_step'] = np.nan
for i, row in df2.iterrows():
    t = row['Temp']
    if t <= 700:  # only impute where we have Figure 1 data
        df2.loc[i, 'seconds_per_extraction_step'] = max(dt_interp(t), 1.0)

print("\n=== Nakhla #2 imputed durations ===")
for _, r in df2.iterrows():
    dt_str = f"{r.seconds_per_extraction_step:.1f}" if not np.isnan(r.seconds_per_extraction_step) else "NaN"
    print(f"{r.Temp:4.0f}°C: dt={dt_str:>8s}s  39Ar={r['39Ar']:.3f}")

# Save
df2_out = df2[['Temp', '39Ar', 'std_39Ar', 'seconds_per_extraction_step']].copy()
df2_out.to_csv('nakhla2_parsed_fitted.csv', index=False)
print(f"\nSaved nakhla2_parsed_fitted.csv ({len(df2_out)} steps, {df2_out.dropna().shape[0]} with durations)")
