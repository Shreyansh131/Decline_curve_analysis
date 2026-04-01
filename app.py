import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# -----------------------------------
# Page config
# -----------------------------------
st.set_page_config(page_title="DCA Dashboard", layout="wide")
st.title("📊 Decline Curve Analysis (DCA) Dashboard")


# -----------------------------------
# Helper functions
# -----------------------------------
def rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.mean((a - b) ** 2))


def exponential(qi, Di, t):
    return qi * np.exp(-Di * t)


def hyperbolic(qi, Di, b, t):
    return qi / ((1 + b * Di * t) ** (1 / b))


def harmonic(qi, Di, t):
    return qi / (1 + Di * t)


def insert_column_beside(df, target_col, new_col_name, new_values):
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe.")

    insert_at = df.columns.get_loc(target_col) + 1

    if new_col_name in df.columns:
        df = df.drop(columns=[new_col_name])
        insert_at = df.columns.get_loc(target_col) + 1

    df.insert(insert_at, new_col_name, new_values)
    return df


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def to_excel_bytes(df, sheet_name="DCA"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


# -----------------------------------
# File upload
# -----------------------------------
file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file is not None:
    # Read file
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # Column selection
    # -----------------------------------
    st.sidebar.header("⚙️ Column Selection")

    well_col = st.sidebar.selectbox("Well ID Column", df.columns)
    date_col = st.sidebar.selectbox("Date Column", df.columns)
    rate_col = st.sidebar.selectbox("Rate Column", df.columns)

    # Convert date safely
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Convert rate safely
    df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
    df = df.dropna(subset=[rate_col])

    # -----------------------------------
    # Well selection
    # -----------------------------------
    st.sidebar.header("🛢️ Well Selection")
    wells = df[well_col].dropna().unique()
    selected_well = st.sidebar.selectbox("Select Well", wells)

    filtered = df[df[well_col] == selected_well].copy()
    filtered = filtered.sort_values(by=date_col).reset_index(drop=True)
    filtered["t"] = np.arange(len(filtered))

    # -----------------------------------
    # DCA parameters
    # -----------------------------------
    st.sidebar.header("📉 DCA Parameters")

    default_qi = float(filtered[rate_col].iloc[0]) if len(filtered) > 0 else 1000.0
    qi = st.sidebar.number_input("Initial Rate (qi)", value=default_qi)
    Di = st.sidebar.slider("Decline Rate (Di)", 0.001, 0.2, 0.02)
    b = st.sidebar.slider("Decline Exponent (b)", 0.01, 1.0, 0.5)

    # -----------------------------------
    # Model selection
    # -----------------------------------
    st.sidebar.header("📌 Select Models")
    show_exp = st.sidebar.checkbox("Exponential", True)
    show_hyp = st.sidebar.checkbox("Hyperbolic", True)
    show_har = st.sidebar.checkbox("Harmonic", True)

    # -----------------------------------
    # Curves
    # -----------------------------------
    t = filtered["t"].values
    actual = filtered[rate_col].values

    exp_curve = exponential(qi, Di, t)
    hyp_curve = hyperbolic(qi, Di, b, t)
    har_curve = harmonic(qi, Di, t)

    # -----------------------------------
    # Metrics
    # -----------------------------------
    results = []
    if show_exp:
        results.append(("Exponential", rmse(actual, exp_curve)))
    if show_hyp:
        results.append(("Hyperbolic", rmse(actual, hyp_curve)))
    if show_har:
        results.append(("Harmonic", rmse(actual, har_curve)))

    # -----------------------------------
    # Plot
    # -----------------------------------
    st.subheader(f"📈 Well: {selected_well}")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(filtered[date_col], actual, label="Actual", linewidth=3)

    if show_exp:
        ax.plot(filtered[date_col], exp_curve, label="Exponential")
    if show_hyp:
        ax.plot(filtered[date_col], hyp_curve, label="Hyperbolic")
    if show_har:
        ax.plot(filtered[date_col], har_curve, label="Harmonic")

    ax.set_xlabel("Date")
    ax.set_ylabel("Production Rate")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # -----------------------------------
    # Metrics table
    # -----------------------------------
    st.subheader("📊 Model Performance (RMSE)")
    if results:
        metrics_df = pd.DataFrame(results, columns=["Model", "RMSE"])
        st.dataframe(metrics_df)

        best_model = metrics_df.loc[metrics_df["RMSE"].idxmin(), "Model"]
        st.success(f"🏆 Best Fit Model: {best_model}")
    else:
        st.warning("Please select at least one model.")

    # -----------------------------------
    # Download section (CLEAN FORMAT)
    # -----------------------------------
    st.subheader("⬇️ Download DCA Data")

    # Create clean dataframe (only required columns)
    base_df = filtered[[well_col, date_col, rate_col]].copy()


    # Add curves beside rate column
    def create_output_df(curve, col_name):
        df_out = base_df.copy()
        insert_at = df_out.columns.get_loc(rate_col) + 1
        df_out.insert(insert_at, col_name, curve)
        return df_out


    exp_df = create_output_df(exp_curve, "DCA_Exp")
    hyp_df = create_output_df(hyp_curve, "DCA_Hyper")
    har_df = create_output_df(har_curve, "DCA_Har")

    # Convert to Excel
    from io import BytesIO


    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()


    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download Exponential",
            data=to_excel(exp_df),
            file_name=f"{selected_well}_DCA_Exp.xlsx"
        )

    with col2:
        st.download_button(
            "Download Hyperbolic",
            data=to_excel(hyp_df),
            file_name=f"{selected_well}_DCA_Hyper.xlsx"
        )

    with col3:
        st.download_button(
            "Download Harmonic",
            data=to_excel(har_df),
            file_name=f"{selected_well}_DCA_Har.xlsx"
        )