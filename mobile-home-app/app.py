from datetime import datetime
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import streamlit as st

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="As-Is Mobile Home Wholesale Valuation Engine",
    page_icon="📱",
    layout="wide",
)

# --- SAVED OFFERS PERSISTENT FILE ---
SAVED_OFFERS_FILE = "saved_offers.csv"


def load_saved_offers():
  if os.path.exists(SAVED_OFFERS_FILE):
    try:
      return pd.read_csv(SAVED_OFFERS_FILE)
    except Exception:
      return pd.DataFrame()
  return pd.DataFrame()


def save_offer_to_file(data):
  df = load_saved_offers()
  new_row = pd.DataFrame([data])
  if not df.empty and "Address" in df.columns:
    # Remove existing record for the same address to update it
    mask = (
        df["Address"].astype(str).str.strip().str.lower()
        == data["Address"].strip().lower()
    )
    df = df[~mask]
  df = pd.concat([df, new_row], ignore_index=True)
  df.to_csv(SAVED_OFFERS_FILE, index=False)
  return df


# --- PASSWORD PROTECTION ---
def check_password():
  if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

  if not st.session_state["password_correct"]:
    st.title("🔒 Login Required")
    password_input = st.text_input("Enter Access Key:", type="password")
    if st.button("Login"):
      if password_input == "BuySCMH2026!":
        st.session_state["password_correct"] = True
        st.rerun()
      else:
        st.error("Incorrect Password")
    return False
  return True


if not check_password():
  st.stop()

# --- APP HEADER ---
st.title("📱 As-Is Mobile Home Wholesale Valuation Engine")
st.markdown(
    "This engine calculates max allowable offers for immediate liquidation"
    " flips based on structural variables, condition scores, and tailored"
    " moving penalties."
)
st.markdown("---")


# --- DYNAMIC FILE PATH RESOLVER ---
def find_data_file(filename="Combined_Sales_Data_2022-2026.csv"):
  base_dir = os.path.dirname(os.path.abspath(__file__))
  possible_paths = [
      os.path.join(base_dir, filename),
      filename,
      os.path.join("mobile-home-app", filename),
  ]
  for path in possible_paths:
    if os.path.exists(path):
      return path
  return None


@st.cache_data
def load_and_clean_data():
  file_path = find_data_file()
  if not file_path:
    return None

  try:
    df = pd.read_csv(file_path)
    if "Sales Price" not in df.columns:
      df = pd.read_csv(file_path, header=1)
  except Exception:
    return None

  def clean_currency(val):
    if pd.isna(val):
      return np.nan
    val_str = str(val).replace("$", "").replace(",", "").strip()
    try:
      return float(val_str)
    except:
      return np.nan

  df["Sales Price Clean"] = df["Sales Price"].apply(clean_currency)
  df_clean = df.dropna(subset=["Sales Price Clean"]).copy()

  df_clean["Year Built Clean"] = pd.to_numeric(
      df_clean["Year Built"], errors="coerce"
  )
  df_clean["Year Built Clean"] = df_clean["Year Built Clean"].fillna(
      df_clean["Year Built Clean"].median()
  )

  def clean_rating(val):
    if pd.isna(val):
      return np.nan
    val_str = str(val).split("/")[0].strip()
    try:
      return float(val_str)
    except:
      return np.nan

  df_clean["Rating Clean"] = df_clean["Rating"].apply(clean_rating)
  df_clean["Rating Clean"] = df_clean["Rating Clean"].fillna(
      df_clean["Rating Clean"].median()
  )

  df_clean["Size_Double"] = (
      df_clean["Size (Single/Double)"]
      .astype(str)
      .str.strip()
      .str.capitalize()
      == "Double"
  ).astype(int)
  df_clean["MustMove_Yes"] = (
      df_clean["Must Move (Yes/No)"].astype(str).str.strip().str.capitalize()
      == "Yes"
  ).astype(int)

  return df_clean


df_dataset = load_and_clean_data()

# --- SIDEBAR: SAVED DEALS LOADER ---
saved_df = load_saved_offers()

st.sidebar.header("📂 Load Saved Property")
saved_addresses = ["-- Select New Property --"]
if not saved_df.empty and "Address" in saved_df.columns:
  saved_addresses += list(saved_df["Address"].dropna().unique())

selected_saved_prop = st.sidebar.selectbox(
    "Pull Up Previous Offer:", saved_addresses
)

# Defaults
def_address = "Columbia, SC"
def_size = "Singlewide"
def_year = 1990
def_beds = 3
def_baths = 2.00
def_move = "Must Be Moved (Yes)"
def_cond = 7
def_risk = 80
def_profit = 0.50

# Pre-fill if saved property selected
if (
    selected_saved_prop != "-- Select New Property --"
    and not saved_df.empty
    and "Address" in saved_df.columns
):
  match_row = saved_df[saved_df["Address"] == selected_saved_prop]
  if not match_row.empty:
    row_data = match_row.iloc[-1]
    def_address = str(row_data.get("Address", def_address))
    def_size = str(row_data.get("Footprint Size", def_size))
    def_year = int(row_data.get("Year Built", def_year))
    def_beds = int(row_data.get("Bedrooms", def_beds))
    def_baths = float(row_data.get("Bathrooms", def_baths))
    def_move = str(row_data.get("Must Move", def_move))
    def_cond = int(row_data.get("Condition Rating", def_cond))
    def_risk = int(row_data.get("Risk Discount (%)", def_risk))
    def_profit = float(row_data.get("Target Profit Ratio", def_profit))

st.sidebar.markdown("---")

# --- MAIN TWO-COLUMN DASHBOARD LAYOUT ---
col_left, col_right = st.columns([1, 1], gap="large")

# ==================== LEFT COLUMN: CONTROL PARAMETERS ====================
with col_left:
  st.subheader("📝 Control Parameters")

  property_address = st.text_input("Property Address", value=def_address)

  size_opts = ["Singlewide", "Doublewide"]
  size_idx = size_opts.index(def_size) if def_size in size_opts else 0
  footprint_size = st.selectbox(
      "Footprint Size Class", size_opts, index=size_idx
  )

  year_built = st.number_input(
      "Year Built", min_value=1970, max_value=2026, value=def_year, step=1
  )

  c_bed, c_bath = st.columns(2)
  bedrooms = c_bed.number_input(
      "Bedrooms", min_value=1, max_value=6, value=def_beds, step=1
  )
  bathrooms = c_bath.number_input(
      "Bathrooms", min_value=1.0, max_value=4.0, value=def_baths, step=0.5
  )

  move_opts = ["Stay Put (No)", "Must Be Moved (Yes)"]
  move_idx = move_opts.index(def_move) if def_move in move_opts else 1
  must_move_radio = st.radio(
      "Logistics Status: Does the home need to be moved?",
      move_opts,
      index=move_idx,
  )

  condition_rating = st.slider(
      "As-Is Condition Rating (1=Total Wreck, 5=Average, 10=Pristine)",
      min_value=1,
      max_value=10,
      value=def_cond,
  )

  risk_discount_pct = (
      st.slider(
          "Risk Discount Modifier Strategy (%)",
          min_value=50,
          max_value=100,
          value=def_risk,
          help="Discount percentage applied to ceiling value.",
      )
      / 100.0
  )

  target_profit_ratio = st.slider(
      "Target Profit per Dollar Invested ($)",
      min_value=0.10,
      max_value=1.00,
      value=def_profit,
      step=0.05,
      help="Required profit ratio relative to investment capital.",
  )

# ==================== UNDERWRITING ENGINE MATH (STATISTICAL MODEL) ====================
base_benchmark = 34715.0 if footprint_size == "Doublewide" else 23431.0
base_year = 2000
age_adjustment = (year_built - base_year) * 483.0
condition_adjustment = (condition_rating - 5) * 4063.0
relocation_penalty = 12951.0 if "Yes" in must_move_radio else 0.0

gross_resale_value = max(
    3000.0,
    base_benchmark
    + age_adjustment
    + condition_adjustment
    - relocation_penalty,
)
liquidation_ceiling = gross_resale_value
target_exit_value = liquidation_ceiling * risk_discount_pct

max_allowable_offer = target_exit_value / (1.0 + target_profit_ratio)
targeted_profit = max_allowable_offer * target_profit_ratio

# ==================== RIGHT COLUMN: LIQUIDATION MATH PROOF ====================
with col_right:
  st.subheader("📈 As-Is Liquidation Math Proof")

  m1, m2 = st.columns(2)
  m1.metric("MAX ALLOWABLE OFFER", f"${max_allowable_offer:,.2f}")
  m2.metric("Targeted Wholesale Profit", f"${targeted_profit:,.2f}")

  st.markdown("---")

  st.subheader("📊 Underwriting Breakdown")

  st.markdown(
      f"* **Footprint Baseline:** Unit recognized as a `{footprint_size}`."
      f" Regression baseline benchmark set to `${base_benchmark:,.2f}`."
  )
  st.markdown(
      f"* **Age Index Adjustment:** Built in **{year_built}**. Adjusted value by"
      f" **${age_adjustment:+,.2f}** ($483/year relative to 2000 baseline)."
  )
  st.markdown(
      f"* **Condition Index Scaling:** Condition score"
      f" **{condition_rating}/10** adjusted raw value by"
      f" **${condition_adjustment:+,.2f}** ($4,063/rating point)."
  )

  if relocation_penalty > 0:
    st.markdown(
        f"* **Logistics Factor:** ⚠️ **-${relocation_penalty:,.2f}** relocation"
        f" discount applied for a {footprint_size} required to move."
    )
  else:
    st.markdown(
        "* **Logistics Factor:** No relocation penalty applied (In-Park / Stays"
        " on lot)."
    )

  st.markdown(
      f"* **Calculated Gross Resale Value:** `${gross_resale_value:,.2f}`"
  )

  st.markdown(
      f"""
    <div style="background-color: #1a2d42; padding: 20px; border-radius: 8px; border: 1px solid #2d4f7c; margin-top: 15px; margin-bottom: 20px;">
        <h4 style="color: #64b5f6; margin-top: 0;">Liquidation Accounting Checklist:</h4>
        <ul style="color: #e0e0e0; list-style-type: disc; padding-left: 20px; line-height: 1.8;">
            <li><b>Adjusted Liquidation Value Ceiling:</b> ${liquidation_ceiling:,.2f}</li>
            <li><b>Risk Discount Strategy Level ({int(risk_discount_pct*100)}%):</b> ${target_exit_value:,.2f}</li>
            <hr style="border-color: #2d4f7c; margin: 10px 0;">
            <li><b>Target Profit Ratio:</b> {target_profit_ratio:.2f} profit for every 1.00 invested</li>
            <li><b>Maximum Acquisition Cost (Your Investment):</b> ${max_allowable_offer:,.2f}</li>
            <li><b>Resulting Minimum Assignment Spread:</b> ${targeted_profit:,.2f}</li>
            <hr style="border-color: #2d4f7c; margin: 10px 0;">
            <li><b>Mathematical Proof:</b> ${max_allowable_offer:,.2f} (Purchase) + ${targeted_profit:,.2f} (Profit) = ${target_exit_value:,.2f} (Target Exit Value)</li>
        </ul>
    </div>
    """,
      unsafe_allow_html=True,
  )

  # SAVE DEAL ACTION
  if st.button("Log Deal Data & Lock Offer", use_container_width=True):
    record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Address": property_address,
        "Footprint Size": footprint_size,
        "Year Built": year_built,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Must Move": must_move_radio,
        "Condition Rating": condition_rating,
        "Risk Discount (%)": int(risk_discount_pct * 100),
        "Target Profit Ratio": target_profit_ratio,
        "Gross Resale Value": round(gross_resale_value, 2),
        "Target Exit Value": round(target_exit_value, 2),
        "MAO": round(max_allowable_offer, 2),
        "Targeted Profit": round(targeted_profit, 2),
    }
    save_offer_to_file(record)
    st.success(
        f"✅ Offer of ${max_allowable_offer:,.2f} for '{property_address}' saved"
        " successfully!"
    )
    st.rerun()

# ==================== PORTFOLIO OF SAVED OFFERS ====================
st.markdown("---")
saved_df_current = load_saved_offers()
if not saved_df_current.empty:
  with st.expander("📁 View Portfolio of Locked & Saved Offers"):
    st.markdown(
        "Here are all previously underwritten properties and locked offers:"
    )

    # Format currency columns
    display_df = saved_df_current.copy()
    for col in [
        "Gross Resale Value",
        "Target Exit Value",
        "MAO",
        "Targeted Profit",
    ]:
      if col in display_df.columns:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")

    st.dataframe(display_df, use_container_width=True)

# ==================== HISTORICAL COMPS LOOKUP SECTION ====================
if df_dataset is not None and len(df_dataset) > 0:
  with st.expander("🔍 View Historical Sales Database (Top Matching Comps)"):
    features = [
        "Size_Double",
        "Year Built Clean",
        "MustMove_Yes",
        "Rating Clean",
    ]
    X = df_dataset[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_scaled)

    is_double = 1 if footprint_size == "Doublewide" else 0
    is_move = 1 if "Yes" in must_move_radio else 0

    subj_df = pd.DataFrame([{
        "Size_Double": is_double,
        "Year Built Clean": float(year_built),
        "MustMove_Yes": is_move,
        "Rating Clean": float(condition_rating),
    }])[features]

    subj_scaled = scaler.transform(subj_df)
    distances, indices = knn.kneighbors(subj_scaled)
    matched_comps = df_dataset.iloc[indices[0]].copy()

    display_cols = [
        "Address",
        "Sales Price Clean",
        "Size (Single/Double)",
        "Year Built Clean",
        "Must Move (Yes/No)",
        "Rating",
        "Sale Date",
    ]
    comps_display = matched_comps[display_cols].rename(
        columns={
            "Sales Price Clean": "Sold Price ($)",
            "Year Built Clean": "Year Built",
            "Size (Single/Double)": "Size",
            "Must Move (Yes/No)": "Must Move",
        }
    )
    comps_display["Sold Price ($)"] = comps_display["Sold Price ($)"].map(
        "${:,.0f}".format
    )
    st.dataframe(comps_display, use_container_width=True)
