import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="SC Mobile Home Valuation Engine", page_icon="🏡", layout="wide")

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

st.title("🏡 Mobile Home Deal Valuation & Offer Calculator")
st.markdown("Use historical South Carolina sales data (2022–2026) to estimate resale value and calculate Maximum Allowable Offers (MAO).")

@st.cache_data
def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    def clean_currency(val):
        if pd.isna(val): return np.nan
        val_str = str(val).replace('$', '').replace(',', '').strip()
        try: return float(val_str)
        except: return np.nan

    df['Sales Price Clean'] = df['Sales Price'].apply(clean_currency)
    df_clean = df.dropna(subset=['Sales Price Clean']).copy()
    df_clean['Year Built Clean'] = pd.to_numeric(df_clean['Year Built'], errors='coerce')
    df_clean['Year Built Clean'] = df_clean['Year Built Clean'].fillna(df_clean['Year Built Clean'].median())

    def clean_rating(val):
        if pd.isna(val): return np.nan
        val_str = str(val).split('/')[0].strip()
        try: return float(val_str)
        except: return np.nan

    df_clean['Rating Clean'] = df_clean['Rating'].apply(clean_rating)
    df_clean['Rating Clean'] = df_clean['Rating Clean'].fillna(df_clean['Rating Clean'].median())
    df_clean['Size_Double'] = (df_clean['Size (Single/Double)'].astype(str).str.strip().str.capitalize() == 'Double').astype(int)
    df_clean['MustMove_Yes'] = (df_clean['Must Move (Yes/No)'].astype(str).str.strip().str.capitalize() == 'Yes').astype(int)
    return df_clean

csv_file = "Combined_Sales_Data_2022-2026.csv"
df_dataset = load_and_clean_data(csv_file)

if df_dataset is None or len(df_dataset) == 0:
    st.error("Missing dataset. Ensure 'Combined_Sales_Data_2022-2026.csv' is in the project folder.")
    st.stop()

st.sidebar.success(f"Loaded {len(df_dataset)} Sales Records")

features = ['Size_Double', 'Year Built Clean', 'MustMove_Yes', 'Rating Clean']
X = df_dataset[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)

st.sidebar.header("📝 Subject Property Specs")
lead_address = st.sidebar.text_input("Property Address", "123 Main St, Moncks Corner, SC")
size_input = st.sidebar.radio("Home Size", ["Single", "Double"])
year_input = st.sidebar.number_input("Year Built", min_value=1970, max_value=2026, value=2002)
must_move_input = st.sidebar.radio("Must Be Moved Off Lot?", ["No (In-Park / Stays)", "Yes (Must Move)"])
condition_rating = st.sidebar.slider("Condition Rating (1-10)", 1, 10, 7)

st.sidebar.header("💰 Deal Financial Parameters")
target_margin_pct = st.sidebar.slider("Target Profit Margin (%)", 10, 40, 25, 5) / 100.0
estimated_rehab = st.sidebar.number_input("Estimated Rehab ($)", min_value=0, value=5000, step=500)
default_move = 3000 if "Yes" in must_move_input else 0
estimated_move = st.sidebar.number_input("Estimated Transport ($)", min_value=0, value=default_move, step=500)

is_double = 1 if size_input == "Double" else 0
is_move = 1 if "Yes" in must_move_input else 0

subject_data = pd.DataFrame([{'Size_Double': is_double, 'Year Built Clean': float(year_input), 'MustMove_Yes': is_move, 'Rating Clean': float(condition_rating)}])[features]
subject_scaled = scaler.transform(subject_data)
distances, indices = knn.kneighbors(subject_scaled)

matched_comps = df_dataset.iloc[indices[0]].copy()
est_resale_price = matched_comps['Sales Price Clean'].mean()
min_comp_price = matched_comps['Sales Price Clean'].min()
max_comp_price = matched_comps['Sales Price Clean'].max()

target_profit = est_resale_price * target_margin_pct
mao_offer = est_resale_price - target_profit - estimated_rehab - estimated_move

st.subheader(f"Valuation Summary for **{lead_address}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estimated Resale Price", f"${est_resale_price:,.0f}")
c2.metric("Recommended Max Offer (MAO)", f"${mao_offer:,.0f}", delta=f"Margin: {target_margin_pct*100:.0f}%")
c3.metric("Target Profit", f"${target_profit:,.0f}")
c4.metric("Market Comp Range", f"${min_comp_price:,.0f} - ${max_comp_price:,.0f}")

st.markdown("---")
st.subheader("📊 Top 5 Comparable Past Sales")
display_cols = ['Address', 'Sales Price Clean', 'Size (Single/Double)', 'Year Built Clean', 'Must Move (Yes/No)', 'Rating', 'Sale Date']
comps_display = matched_comps[display_cols].rename(columns={'Sales Price Clean': 'Sold Price ($)', 'Year Built Clean': 'Year Built'})
comps_display['Sold Price ($)'] = comps_display['Sold Price ($)'].map('${:,.0f}'.format)
st.dataframe(comps_display, use_container_width=True)
