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
    "This engine calculates max allowable offer ranges for immediate"
    " liquidation flips based on structural variables, condition scores, and"
    " tailored moving penalties."
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

# --- MAIN TWO-COLUMN DASHBOARD LAYOUT ---
col_left, col_right = st.columns([1, 1], gap="large")

# ==================== LEFT COLUMN: CONTROL PARAMETERS ====================
with col_left:
  st.subheader("📝 Control Parameters")

  property_address = st.text_input("Property Address", value="Columbia, SC")

  footprint_size = st.selectbox(
      "Footprint Size Class", ["Doublewide", "Singlewide"]
  )

  year_built = st.number_input(
      "Year Built", min_value=1970, max_value=2026, value=2017, step=1
  )

  c_bed, c_bath = st.columns(2)
  bedrooms = c_bed.number_input(
      "Bedrooms", min_value=1, max_value=6, value=4, step=1
  )
  bathrooms = c_bath.number_input(
      "Bathrooms", min_value=1.0, max_value=4.0, value=2.00, step=0.5
  )

  must_move_radio = st.radio(
      "Logistics Status: Does the home need to be moved?",
      ["Stay Put (No)", "Must Be Moved (Yes)"],
      index=1,
  )

  condition_rating = st.slider(
      "As-Is Condition Rating (1=Total Wreck, 5=Average, 10=Pristine)",
      min_value=1,
      max_value=10,
      value=10,
  )

  risk_discount_pct = (
      st.slider(
          "Risk Discount Modifier Strategy (%)",
          min_value=50,
          max_value=100,
          value=80,
          help="Discount percentage applied to ceiling value.",
      )
      / 100.0
  )

  target_profit_ratio = st.slider(
      "Target Profit per Dollar Invested ($)",
      min_value=0.10,
      max_value=1.00,
      value=0.50,
      step=0.05,
      help="Required profit ratio relative to investment capital.",
  )

  # NEW: Offer Range Spread Control
  offer_spread = st.slider(
      "Negotiation Buffer Below Ceiling ($)",
      min_value=3000,
      max_value=5000,
      value=5000,
      step=500,
      help=(
          "Amount subtracted from the calculated MAO ceiling to set your"
          " starting low-end offer."
      ),
  )

# ==================== UNDERWRITING ENGINE MATH ====================
base_benchmark = 50000.0 if footprint_size == "Doublewide" else 30000.0
base_year = 2000
age_adjustment = (year_built - base_year) * 500.0
condition_adjustment = (condition_rating - 5) * 3500.0

gross_resale_value = base_benchmark + age_adjustment + condition_adjustment

relocation_penalty = 0.0
if "Yes" in must_move_radio:
  relocation_penalty = 10000.0 if footprint_size == "Doublewide" else 5000.0

liquidation_ceiling = gross_resale_value - relocation_penalty
target_exit_value = liquidation_ceiling * risk_discount_pct

# MAO High-End Ceiling
max_allowable_offer_high = target_exit_value / (1.0 + target_profit_ratio)
# Low-End Offer Anchor
max_allowable_offer_low = max_allowable_offer_high - offer_spread

targeted_profit = max_allowable_offer_high * target_profit_ratio

# ==================== RIGHT COLUMN: LIQUIDATION MATH PROOF ====================
with col_right:
  st.subheader("📈 As-Is Liquidation Math Proof")

  m1, m2 = st.columns(2)
  m1.metric(
      "RECOMMENDED OFFER RANGE",
      f"${max_allowable_offer_low:,.2f} – ${max_allowable_offer_high:,.2f}",
  )
  m2.metric("Targeted Wholesale Profit", f"${targeted_profit:,.2f}")

  st.markdown("---")

  st.subheader("📊 Underwriting Breakdown")

  st.markdown(
      f"* **Footprint Starting Point:** Unit recognized as a `{footprint_size}`"
      f" . Initial median benchmark value set to `${base_benchmark:,.2f}`."
  )
  st.markdown(
      f"* **Age Index Adjustment:** Built in **{year_built}**. Scaled value by"
      f" **+${age_adjustment:,.2f}** against market year baseline."
  )
  st.markdown(
      f"* **Condition Index Scaling:** Condition score"
      f" **{condition_rating}/10** adjusted raw asset value by"
      f" **+${condition_adjustment:,.2f}** directly at the baseline."
  )

  if relocation_penalty > 0:
    st.markdown(
        f"* **Logistics Factor:** ⚠️ **-${relocation_penalty:,.2f}** relocation"
        f" penalty applied for a {footprint_size} required to move."
    )
  else:
    st.markdown(
        "* **Logistics Factor:** No relocation penalty applied (sits in park"
        " / stays on lot)."
    )

  st.markdown(
      f"* **Calculated Gross Resale Value:** `${gross_resale_value:,.2f}`"
  )

  # Styled Blue Info Container Box
  st.markdown(
      f"""
    <div style="background-color: #1a2d42; padding: 20px; border-radius: 8px; border: 1px solid #2d4f7c; margin-top: 15px; margin-bottom: 20px;">
        <h4 style="color: #64b5f6; margin-top: 0;">Liquidation Accounting Checklist:</h4>
        <ul style="color: #e0e0e0; list-style-type: disc; padding-left: 20px; line-height: 1.8;">
            <li><b>Adjusted Liquidation Value Ceiling:</b> ${liquidation_ceiling:,.2f}</li>
            <li><b>Risk Discount Strategy Level ({int(risk_discount_pct*100)}%):</b> ${target_exit_value:,.2f}</li>
            <hr style="border-color: #2d4f7c; margin: 10px 0;">
            <li><b>Target Profit Ratio:</b> {target_profit_ratio:.2f} profit for every 1.00 invested</li>
            <li><b>Initial Opening Offer (Low Anchor):</b> ${max_allowable_offer_low:,.2f}</li>
            <li><b>Maximum Acquisition Ceiling (High End):</b> ${max_allowable_offer_high:,.2f}</li>
            <li><b>Resulting Minimum Assignment Spread:</b> ${targeted_profit:,.2f}</li>
            <hr style="border-color: #2d4f7c; margin: 10px 0;">
            <li><b>Mathematical Proof:</b> ${max_allowable_offer_high:,.2f} (Max Purchase) + ${targeted_profit:,.2f} (Profit) = ${target_exit_value:,.2f} (Target Exit Value)</li>
        </ul>
    </div>
    """,
      unsafe_allow_html=True,
  )

  if st.button("Log Deal Data & Lock Offer Range", use_container_width=False):
    st.success(
        f"Offer range of ${max_allowable_offer_low:,.2f} –"
        f" ${max_allowable_offer_high:,.2f} locked for {property_address}!"
    )

# ==================== HISTORICAL COMPS LOOKUP SECTION ====================
if df_dataset is not None and len(df_dataset) > 0:
  st.markdown("---")
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
