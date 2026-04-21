# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:03:47 2026

@author: Weihua Huang
"""
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================================
# Paths
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
MODEL_DIR = os.path.join(BASE_DIR, "models")
TRAIN_DIR = os.path.join(BASE_DIR, "train_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

INPUT_FILE = os.path.join(INPUT_DIR, "input_samples.xlsx")
PT_TRAIN_FILE = os.path.join(TRAIN_DIR, "df_P_T_train.xlsx")
FO2_TRAIN_FILE = os.path.join(TRAIN_DIR, "df_fO2_train.xlsx")

P_MODEL_FILE = os.path.join(MODEL_DIR, "P_model_tabpfn.pkl")
T_MODEL_FILE = os.path.join(MODEL_DIR, "T_model_tabpfn.pkl")
FO2_MODEL_FILE = os.path.join(MODEL_DIR, "fO2_model_tabpfn.pkl")

# ==========================================================
# Plot style
# ==========================================================
COLOR_ID = "#4C72B0"
COLOR_PT_OOD = "#2CA02C"
COLOR_FEATURE_OOD = "#D62728"
COLOR_FEATURE_RANGE_OOD = "#8C564B"

def set_sa_style():
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 100,
        "savefig.dpi": 600,
    })

def beautify_axis(ax, spine_width=1.1):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        width=1.2,
        length=4,
        bottom=True,
        top=False,
        left=True,
        right=False
    )

# ==========================================================
# Chemistry
# ==========================================================
molar_masses = {
    "SiO2": 60.08,
    "TiO2": 79.87,
    "Al2O3": 101.96,
    "FeO": 71.84,
    "MgO": 40.30,
    "CaO": 56.08,
    "Na2O": 61.98,
    "Cr2O3": 151.99,
    "MnO": 70.94,
}

cation_numbers = {
    "Si": 1,
    "Ti": 1,
    "Al": 2,
    "Fe": 1,
    "Mg": 1,
    "Ca": 1,
    "Na": 2,
    "Cr": 2,
    "Mn": 1,
}

def calculate_chemical_formula(df: pd.DataFrame) -> pd.DataFrame:
    oxygen_atoms = 12
    results = []

    for _, row in df.iterrows():
        n_Si = row["Si_Grt"] / molar_masses["SiO2"]
        n_Ti = row["Ti_Grt"] / molar_masses["TiO2"]
        n_Al = row["Al_Grt"] / molar_masses["Al2O3"]
        n_Fe = row["Fe_Grt"] / molar_masses["FeO"]
        n_Mg = row["Mg_Grt"] / molar_masses["MgO"]
        n_Ca = row["Ca_Grt"] / molar_masses["CaO"]
        n_Na = row["Na_Grt"] / molar_masses["Na2O"]
        n_Cr = row["Cr_Grt"] / molar_masses["Cr2O3"]
        n_Mn = row["Mn_Grt"] / molar_masses["MnO"]

        total_oxygen = (
            n_Si * 2 + n_Ti * 2 + n_Al * 3 + n_Fe + n_Mg + n_Ca + n_Na + n_Cr * 3 + n_Mn
        )

        z = oxygen_atoms / total_oxygen

        ion_numbers = {
            "Si": n_Si * cation_numbers["Si"] * z,
            "Ti": n_Ti * cation_numbers["Ti"] * z,
            "Al": n_Al * cation_numbers["Al"] * z,
            "Fe": n_Fe * cation_numbers["Fe"] * z,
            "Mg": n_Mg * cation_numbers["Mg"] * z,
            "Ca": n_Ca * cation_numbers["Ca"] * z,
            "Na": n_Na * cation_numbers["Na"] * z,
            "Cr": n_Cr * cation_numbers["Cr"] * z,
            "Mn": n_Mn * cation_numbers["Mn"] * z,
        }

        results.append({
            "Si": ion_numbers["Si"],
            "Ti": ion_numbers["Ti"],
            "Al": ion_numbers["Al"],
            "Cr": ion_numbers["Cr"],
            "Fe": ion_numbers["Fe"],
            "Mn": ion_numbers["Mn"],
            "Mg": ion_numbers["Mg"],
            "Ca": ion_numbers["Ca"],
            "Na": ion_numbers["Na"],
            "Correction_Factor_z": z,
        })

    return pd.DataFrame(results)

def garnet_cation_filter(df_input: pd.DataFrame) -> pd.DataFrame:
    oxide_cols = [
        "Si_Grt", "Ti_Grt", "Al_Grt", "Cr_Grt", "Fe_Grt",
        "Mn_Grt", "Mg_Grt", "Ca_Grt", "Na_Grt",
    ]
    other_cols = [c for c in df_input.columns if c not in oxide_cols]

    formula_df = calculate_chemical_formula(df_input[oxide_cols])
    if "Correction_Factor_z" in formula_df.columns:
        formula_df = formula_df.drop(columns=["Correction_Factor_z"])

    formula_df.columns = [c if c.endswith("_Grt") else f"{c}_Grt" for c in formula_df.columns]
    origin_df = df_input[oxide_cols].add_suffix("_origin")
    other_df = df_input[other_cols]

    df = pd.concat([formula_df, origin_df, other_df], axis=1)
    df = df.rename(columns={
        "Y value_P": "P",
        "Y value_T": "T",
        "Y value_fO2": "fO2",
    })

    cation_cols = [
        "Si_Grt", "Ti_Grt", "Al_Grt", "Cr_Grt", "Fe_Grt",
        "Mn_Grt", "Mg_Grt", "Ca_Grt", "Na_Grt",
    ]
    df["cation_sum"] = df[cation_cols].sum(axis=1)

    original_count = len(df)
    df = df[(df["cation_sum"] >= 7.9) & (df["cation_sum"] <= 8.1)].copy()
    removed_count = original_count - len(df)
    print(f"Removed samples (cation filter): {removed_count}")

    return df

# ==========================================================
# Thermodynamic buffers
# ==========================================================
def calculate_iw(P, T):
    P = np.asarray(P)
    T = np.asarray(T)
    T_kelvin = T + 273.15

    x0 = -18.64
    x1 = 0.04359
    x2 = -5.069e-6

    P_transition = x0 + x1 * T_kelvin + x2 * T_kelvin**2

    a = 6.844864 + 1.175691e-1 * P + 1.143873e-3 * P**2
    b = 5.791364e-4 - 2.891434e-4 * P - 2.737171e-7 * P**2
    c = -7.971469e-5 + 3.198005e-5 * P + 1.059554e-10 * P**3 + 2.014461e-7 * np.sqrt(P)
    d = -2.769002e4 + 5.285977e2 * P - 2.919275 * P**2
    iw_fcc = a + b * T_kelvin + c * T_kelvin * np.log(T_kelvin) + d / T_kelvin

    e = 8.463095 - 3.000307e-3 * P + 7.213445e-5 * P**2
    f = 1.148738e-3 - 9.352312e-5 * P + 5.161592e-7 * P**2
    g = -7.448624e-4 - 6.329325e-6 * P - 1.407339e-10 * P**3 + 1.830014e-4 * np.sqrt(P)
    h = -2.782082e4 + 5.285977e2 * P - 8.473231e-1 * P**2
    iw_hcp = e + f * T_kelvin + g * T_kelvin * np.log(T_kelvin) + h / T_kelvin

    return np.where(P <= P_transition, iw_fcc, iw_hcp)

def fmq_hp2011(P, T):
    P = np.asarray(P)
    T = np.asarray(T)
    T_kelvin = T + 273.15
    P_kbar = P * 10.0

    numerator = (
        -587474.0
        + 1584.427 * T_kelvin
        - 203.3164 * T_kelvin * np.log(T_kelvin)
        + 0.09271 * T_kelvin**2
        + 1810.0 * P_kbar
    )
    denominator = 8.3144 * T_kelvin * np.log(10.0)
    return numerator / denominator

def nno_campbell2009(P, T):
    P = np.asarray(P)
    T = np.asarray(T)
    T_kelvin = T + 273.15

    a0, a1, a2, a3, a4 = 8.699000, 0.016420, -0.000276, 2.6830e-06, -1.0150e-08
    b0, b1, b2, b3 = -24205.0, 444.730000, -0.592880, 0.001529

    return (
        a0 + a1 * P + a2 * P**2 + a3 * P**3 + a4 * P**4
        + (b0 + b1 * P + b2 * P**2 + b3 * P**3) / T_kelvin
    )

def mh_schwab1981(P, T):
    P = np.asarray(P)
    T = np.asarray(T)
    T_kelvin = T + 273.15
    return 14.26 - 24949.0 / T_kelvin + (200.0 * P) / T_kelvin - 0.05 * P

# ==========================================================
# OOD
# ==========================================================
BASE_FEATURES = [
    "Si_Grt", "Ti_Grt", "Al_Grt", "Cr_Grt",
    "Fe_Grt", "Mg_Grt", "Ca_Grt", "Na_Grt",
]

FEATURE_OOD_K = 3
FEATURE_OOD_PERCENTILE = 99
PT_OOD_K = 3
PT_OOD_PERCENTILE = 95
RANDOM_STATE = 42

def build_feature_range_summary(train_full_df, features):
    records = []
    for feature in features:
        s = train_full_df[feature].dropna()
        records.append({
            "feature": feature,
            "q01": s.quantile(0.01),
            "q99": s.quantile(0.99),
            "min": s.min(),
            "max": s.max(),
        })
    return pd.DataFrame(records)

def apply_ood_and_predict(df_predict, df_P_T_train, df_fO2_train, P_model, T_model, fO2_model):
    predict_df = df_predict.copy()
    train_full_df = pd.concat([df_P_T_train, df_fO2_train], ignore_index=True)

    # feature range
    feature_range_df = build_feature_range_summary(train_full_df, BASE_FEATURES)
    feature_range_masks = []

    for _, row in feature_range_df.iterrows():
        feature = row["feature"]
        q01 = row["q01"]
        q99 = row["q99"]

        mask = (predict_df[feature] < q01) | (predict_df[feature] > q99)
        predict_df[f"{feature}_out_of_range"] = mask
        feature_range_masks.append(mask.to_numpy())

    predict_df["is_feature_range_OOD"] = np.any(np.column_stack(feature_range_masks), axis=1)
    predict_df["feature_range_OOD_level"] = np.where(
        predict_df["is_feature_range_OOD"], "Feature range OOD", "In-Distribution"
    )

    # feature-space OOD
    X_train = train_full_df[BASE_FEATURES].copy()
    X_predict = predict_df[BASE_FEATURES].copy()

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_predict_scaled = feature_scaler.transform(X_predict)

    feature_nn = NearestNeighbors(n_neighbors=FEATURE_OOD_K)
    feature_nn.fit(X_train_scaled)
    feature_distances = feature_nn.kneighbors(X_predict_scaled)[0].mean(axis=1)

    feature_train_nn = NearestNeighbors(n_neighbors=FEATURE_OOD_K + 1)
    feature_train_nn.fit(X_train_scaled)
    feature_train_knn_dist = feature_train_nn.kneighbors(X_train_scaled)[0][:, 1:].mean(axis=1)

    feature_threshold_ood = np.percentile(feature_train_knn_dist, FEATURE_OOD_PERCENTILE)
    predict_df["feature_OOD_distance"] = feature_distances
    predict_df["is_feature_OOD"] = feature_distances > feature_threshold_ood
    predict_df["feature_OOD_level"] = np.where(
        predict_df["is_feature_OOD"], "Feature OOD", "In-Distribution"
    )

    # predict P,T
    X_pred = predict_df[BASE_FEATURES].copy()
    predict_df["P"] = P_model.predict(X_pred)
    predict_df["T"] = T_model.predict(X_pred)

    # PT OOD
    PT_train_df = pd.concat(
        [
            df_P_T_train[["P", "T"]].copy(),
            df_fO2_train[["P", "T"]].copy(),
        ],
        ignore_index=True,
    ).dropna()

    PT_train = PT_train_df[["P", "T"]].to_numpy()
    PT_predict = predict_df[["P", "T"]].to_numpy()

    pt_scaler = StandardScaler()
    PT_train_scaled = pt_scaler.fit_transform(PT_train)
    PT_predict_scaled = pt_scaler.transform(PT_predict)

    pt_nn = NearestNeighbors(n_neighbors=PT_OOD_K)
    pt_nn.fit(PT_train_scaled)
    pt_distances = pt_nn.kneighbors(PT_predict_scaled)[0].mean(axis=1)

    pt_train_nn = NearestNeighbors(n_neighbors=PT_OOD_K + 1)
    pt_train_nn.fit(PT_train_scaled)
    pt_train_knn_dist = pt_train_nn.kneighbors(PT_train_scaled)[0][:, 1:].mean(axis=1)

    pt_threshold_ood = np.percentile(pt_train_knn_dist, PT_OOD_PERCENTILE)
    predict_df["PT_OOD_distance"] = pt_distances
    predict_df["is_PT_OOD"] = pt_distances > pt_threshold_ood
    predict_df["PT_OOD_level"] = np.where(
        predict_df["is_PT_OOD"], "P-T OOD", "In-Distribution"
    )

    # IW and fO2
    predict_df["IW"] = calculate_iw(predict_df["P"], predict_df["T"])

    # IW and fO2
    predict_df["IW"] = calculate_iw(predict_df["P"], predict_df["T"])

    X_fO2 = predict_df[BASE_FEATURES + ["IW"]].copy()

    predict_df["logfO2"] = fO2_model.predict(X_fO2)


    # other buffers
    predict_df["FMQ"] = fmq_hp2011(predict_df["P"], predict_df["T"])
    predict_df["NNO"] = nno_campbell2009(predict_df["P"], predict_df["T"])
    predict_df["MH"] = mh_schwab1981(predict_df["P"], predict_df["T"])

    predict_df["dIW"] = predict_df["logfO2"] - predict_df["IW"]
    predict_df["dFMQ"] = predict_df["logfO2"] - predict_df["FMQ"]
    predict_df["dNNO"] = predict_df["logfO2"] - predict_df["NNO"]
    predict_df["dMH"] = predict_df["logfO2"] - predict_df["MH"]

    # final OOD label
    predict_df["OOD类别"] = "In-Distribution"
    predict_df.loc[predict_df["is_PT_OOD"], "OOD类别"] = "P-T OOD"
    predict_df.loc[predict_df["is_feature_OOD"], "OOD类别"] = "Feature OOD"
    predict_df.loc[predict_df["is_feature_range_OOD"], "OOD类别"] = "Feature range OOD"

    return predict_df, feature_range_df, X_train_scaled, X_predict_scaled

# ==========================================================
# PCA plot
# ==========================================================
def save_pca_plot(X_train_scaled, X_predict_scaled, predict_df):
    set_sa_style()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_predict_pca = pca.transform(X_predict_scaled)

    pc1_var = pca.explained_variance_ratio_[0] * 100
    pc2_var = pca.explained_variance_ratio_[1] * 100

    color_map = {
        "In-Distribution": COLOR_ID,
        "P-T OOD": COLOR_PT_OOD,
        "Feature OOD": COLOR_FEATURE_OOD,
        "Feature range OOD": COLOR_FEATURE_RANGE_OOD,
    }

    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    ax.scatter(
        X_train_pca[:, 0], X_train_pca[:, 1],
        alpha=0.55, color="#9A9A9A", s=22, edgecolors="none", label="Training set"
    )

    for category in ["In-Distribution", "P-T OOD", "Feature OOD", "Feature range OOD"]:
        mask = predict_df["OOD类别"] == category
        if np.any(mask):
            ax.scatter(
                X_predict_pca[mask, 0],
                X_predict_pca[mask, 1],
                alpha=0.90,
                color=color_map[category],
                s=36,
                edgecolors="black" if category != "In-Distribution" else "none",
                linewidths=0.35 if category != "In-Distribution" else 0,
                label=category,
            )

    ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax.set_title("PCA of feature space", pad=6)
    beautify_axis(ax)
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "prediction_PCA.png"), bbox_inches="tight")
    plt.savefig(os.path.join(IMAGE_DIR, "prediction_PCA.pdf"), bbox_inches="tight")
    plt.close()

# ==========================================================
# Main
# ==========================================================
def main():
    print("Loading input and training data...")

    for path in [INPUT_FILE, PT_TRAIN_FILE, FO2_TRAIN_FILE, P_MODEL_FILE, T_MODEL_FILE, FO2_MODEL_FILE]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    df_input_raw = pd.read_excel(INPUT_FILE, engine="openpyxl")
    df_P_T_train_raw = pd.read_excel(PT_TRAIN_FILE, engine="openpyxl")
    df_fO2_train_raw = pd.read_excel(FO2_TRAIN_FILE, engine="openpyxl")

    for df_ in [df_input_raw, df_P_T_train_raw, df_fO2_train_raw]:
        df_.fillna(0.00001, inplace=True)

    print("Applying garnet cation filter...")
    df_input = garnet_cation_filter(df_input_raw)
    df_P_T_train = garnet_cation_filter(df_P_T_train_raw)
    df_fO2_train = garnet_cation_filter(df_fO2_train_raw)

    print("Loading models...")
    P_model = joblib.load(P_MODEL_FILE)
    T_model = joblib.load(T_MODEL_FILE)
    fO2_model = joblib.load(FO2_MODEL_FILE)

    print("Running prediction pipeline...")
    predict_df, feature_range_df, X_train_scaled, X_predict_scaled = apply_ood_and_predict(
        df_predict=df_input,
        df_P_T_train=df_P_T_train,
        df_fO2_train=df_fO2_train,
        P_model=P_model,
        T_model=T_model,
        fO2_model=fO2_model,
    )

    save_pca_plot(X_train_scaled, X_predict_scaled, predict_df)

    preferred_cols = []
    for c in ["Sample", "Sample_ID", "sample", "sample_id", "Name", "name"]:
        if c in predict_df.columns:
            preferred_cols.append(c)

    output_cols = (
        preferred_cols
        + BASE_FEATURES
        + [c for c in ["Mn_Grt"] if c in predict_df.columns]
        + ["P", "T", "logfO2", "IW", "FMQ", "NNO", "MH", "dIW", "dFMQ", "dNNO", "dMH", "OOD类别"]
    )

    output_cols = [c for c in output_cols if c in predict_df.columns]
    output_df = predict_df[output_cols].copy()

    output_file = os.path.join(OUTPUT_DIR, "prediction_results.xlsx")
    feature_range_file = os.path.join(OUTPUT_DIR, "feature_range_summary.xlsx")

    output_df.to_excel(output_file, index=False)
    feature_range_df.to_excel(feature_range_file, index=False)

    print("Done.")
    print("Saved:", output_file)
    print("Saved:", feature_range_file)
    print("Saved:", os.path.join(IMAGE_DIR, "prediction_PCA.png"))
    print("Saved:", os.path.join(IMAGE_DIR, "prediction_PCA.pdf"))

if __name__ == "__main__":
    main()
