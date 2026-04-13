import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Tokenized Capital Markets Propensity Insights", 
    page_icon="©️", 
    layout="wide"
)

st.markdown("""
    <style>
        h1, h2, h3 { font-family: 'Inter', sans-serif; }
        [data-testid="stMetricValue"] { color: #2C3E50 !important; font-weight: 800; font-size: 1.8rem;}
        .stTabs [aria-selected="true"] {
            background-color: #F0F4F8;
            border-bottom: 3px solid #2C3E50 !important;
            color: #2C3E50 !important;
            font-weight: bold;
        }
        .snapshot-box {
            background-color: #F8F9FA;
            border: 1px solid #E0E5EC;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            height: 100%;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 2px solid #E0E5EC;
            font-size: 0.95rem;
            color: #555;
        }
        .model-toggle {
            background-color: #F0F4F8;
            padding: 15px 25px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 5px solid #2C3E50;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER DICTIONARIES (FLAGS & COLORS)
# ==========================================
iso3_to_iso2 = {
    'AFG': 'AF', 'ALB': 'AL', 'DZA': 'DZ', 'AND': 'AD', 'AGO': 'AO', 'ARG': 'AR', 'ARM': 'AM', 'AUS': 'AU', 'AUT': 'AT', 'AZE': 'AZ',
    'BHS': 'BS', 'BHR': 'BH', 'BGD': 'BD', 'BRB': 'BB', 'BLR': 'BY', 'BEL': 'BE', 'BLZ': 'BZ', 'BEN': 'BJ', 'BTN': 'BT', 'BOL': 'BO',
    'BIH': 'BA', 'BWA': 'BW', 'BRA': 'BR', 'BRN': 'BN', 'BGR': 'BG', 'BFA': 'BF', 'BDI': 'BI', 'CPV': 'CV', 'KHM': 'KH', 'CMR': 'CM',
    'CAN': 'CA', 'CAF': 'CF', 'TCD': 'TD', 'CHL': 'CL', 'CHN': 'CN', 'COL': 'CO', 'COM': 'KM', 'COG': 'CG', 'COD': 'CD', 'CRI': 'CR',
    'CIV': 'CI', 'HRV': 'HR', 'CUB': 'CU', 'CYP': 'CY', 'CZE': 'CZ', 'DNK': 'DK', 'DJI': 'DJ', 'DMA': 'DM', 'DOM': 'DO', 'ECU': 'EC',
    'EGY': 'EG', 'SLV': 'SV', 'GNQ': 'GQ', 'ERI': 'ER', 'EST': 'EE', 'SWZ': 'SZ', 'ETH': 'ET', 'FJI': 'FJ', 'FIN': 'FI', 'FRA': 'FR',
    'GAB': 'GA', 'GMB': 'GM', 'GEO': 'GE', 'DEU': 'DE', 'GHA': 'GH', 'GRC': 'GR', 'GRD': 'GD', 'GTM': 'GT', 'GIN': 'GN', 'GNB': 'GW',
    'GUY': 'GY', 'HTI': 'HT', 'HND': 'HN', 'HUN': 'HU', 'ISL': 'IS', 'IND': 'IN', 'IDN': 'ID', 'IRN': 'IR', 'IRQ': 'IQ', 'IRL': 'IE',
    'ISR': 'IL', 'ITA': 'IT', 'JAM': 'JM', 'JPN': 'JP', 'JOR': 'JO', 'KAZ': 'KZ', 'KEN': 'KE', 'KIR': 'KI', 'KWT': 'KW', 'KGZ': 'KG',
    'LAO': 'LA', 'LVA': 'LV', 'LBN': 'LB', 'LSO': 'LS', 'LBR': 'LR', 'LBY': 'LY', 'LIE': 'LI', 'LTU': 'LT', 'LUX': 'LU', 'MDG': 'MG',
    'MWI': 'MW', 'MYS': 'MY', 'MDV': 'MV', 'MLI': 'ML', 'MLT': 'MT', 'MRT': 'MR', 'MUS': 'MU', 'MEX': 'MX', 'MDA': 'MD', 'MCO': 'MC',
    'MNG': 'MN', 'MNE': 'ME', 'MAR': 'MA', 'MOZ': 'MZ', 'MMR': 'MM', 'NAM': 'NA', 'NRU': 'NR', 'NPL': 'NP', 'NLD': 'NL', 'NZL': 'NZ',
    'NIC': 'NI', 'NER': 'NE', 'NGA': 'NG', 'PRK': 'KP', 'MKD': 'MK', 'NOR': 'NO', 'OMN': 'OM', 'PAK': 'PK', 'PLW': 'PW', 'PAN': 'PA',
    'PNG': 'PG', 'PRY': 'PY', 'PER': 'PE', 'PHL': 'PH', 'POL': 'PL', 'PRT': 'PT', 'QAT': 'QA', 'ROU': 'RO', 'RUS': 'RU', 'RWA': 'RW',
    'SAU': 'SA', 'SEN': 'SN', 'SRB': 'RS', 'SYC': 'SC', 'SLE': 'SL', 'SGP': 'SG', 'SVK': 'SK', 'SVN': 'SI', 'SLB': 'SB', 'SOM': 'SO', 
    'ZAF': 'ZA', 'KOR': 'KR', 'SSD': 'SS', 'ESP': 'ES', 'LKA': 'LK', 'SDN': 'SD', 'SUR': 'SR', 'SWE': 'SE', 'CHE': 'CH', 'SYR': 'SY', 
    'TJK': 'TJ', 'TZA': 'TZ', 'THA': 'TH', 'TLS': 'TL', 'TGO': 'TG', 'TON': 'TO', 'TTO': 'TT', 'TUN': 'TN', 'TUR': 'TR', 'TKM': 'TM', 
    'UGA': 'UG', 'UKR': 'UA', 'ARE': 'AE', 'GBR': 'GB', 'USA': 'US', 'URY': 'UY', 'UZB': 'UZ', 'VUT': 'VU', 'VEN': 'VE', 'VNM': 'VN', 
    'YEM': 'YE', 'ZMB': 'ZM', 'ZWE': 'ZW', 'HKG': 'HK', 'MAC': 'MO', 'TWN': 'TW', 'PRI': 'PR', 'PSE': 'PS'
}

def get_flag(iso3):
    iso2 = iso3_to_iso2.get(iso3)
    if not iso2: return "🌐" 
    return chr(ord(iso2[0]) + 127397) + chr(ord(iso2[1]) + 127397)

color_map = {
    "Grassroot Adopters": "#e74c3c",   
    "Leapfroggers": "#2ecc71",            
    "Low Demand Economies": "#9b59b6",    
    "Tokenization Hubs": "#3498db"                   
}

# ==========================================
# 3. DATA LOADING PIPELINE (Cached)
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "df_unbrokered_extended.csv")
    df = pd.read_csv(csv_path)
    
    col_mapping = {'Archetype_Label': 'Archetype'}
    for col in df.columns:
        if 'Financial_Closedness' in col: col_mapping[col] = 'Financial_Closedness'
        elif 'Inflation' in col: col_mapping[col] = 'Inflation'
        elif 'Crypto_Adoption_Rank' in col: col_mapping[col] = 'Crypto_Adoption_Rank'
    df = df.rename(columns=col_mapping)
    
    # Standardize Core Variables
    pca_features = ['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank']
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(df[pca_features])
    
    # ---------------------------------------------------------
    # CALCULATE BASELINE PCA MODEL (Perfected Decimal Weights)
    # ---------------------------------------------------------
    wc_pca, wi_pca, wa_pca = 0.525, 0.007, 0.468
    pca_raw = (X_scaled[:, 0] * wc_pca) + (X_scaled[:, 1] * wi_pca) - (X_scaled[:, 2] * wa_pca)
    df['PCA_Index_Score'] = ((pca_raw - pca_raw.min()) / (pca_raw.max() - pca_raw.min())) * 100

    scaler_cluster_pca = StandardScaler()
    cluster_scaled_pca = scaler_cluster_pca.fit_transform(df[['regulation', 'PCA_Index_Score']])
    kmeans_pca = KMeans(n_clusters=4, random_state=42, n_init=10).fit(cluster_scaled_pca)
    
    centers_pca = scaler_cluster_pca.inverse_transform(kmeans_pca.cluster_centers_)
    ord_centers_pca = [(i, cx, cy) for i, (cx, cy) in enumerate(centers_pca)]
    ord_centers_pca.sort(key=lambda x: x[2], reverse=True) 
    top_2, bottom_2 = ord_centers_pca[:2], ord_centers_pca[2:]
    top_2.sort(key=lambda x: x[1])
    bottom_2.sort(key=lambda x: x[1])

    cluster_map_pca = {top_2[0][0]: "Grassroot Adopters", top_2[1][0]: "Leapfroggers", bottom_2[0][0]: "Low Demand Economies", bottom_2[1][0]: "Tokenization Hubs"}
    df['PCA_Archetype'] = kmeans_pca.predict(cluster_scaled_pca)
    df['PCA_Archetype'] = df['PCA_Archetype'].map(cluster_map_pca)

    # ---------------------------------------------------------
    # CALCULATE BASELINE EQUAL WEIGHT MODEL 
    # ---------------------------------------------------------
    equal_raw = (X_scaled[:, 0] * 0.33333) + (X_scaled[:, 1] * 0.33333) - (X_scaled[:, 2] * 0.33333)
    df['Equal_Index_Score'] = ((equal_raw - equal_raw.min()) / (equal_raw.max() - equal_raw.min())) * 100

    scaler_cluster_eq = StandardScaler()
    cluster_scaled_eq = scaler_cluster_eq.fit_transform(df[['regulation', 'Equal_Index_Score']])
    kmeans_eq = KMeans(n_clusters=4, random_state=42, n_init=10).fit(cluster_scaled_eq)
    
    centers_eq = scaler_cluster_eq.inverse_transform(kmeans_eq.cluster_centers_)
    ord_centers_eq = [(i, cx, cy) for i, (cx, cy) in enumerate(centers_eq)]
    ord_centers_eq.sort(key=lambda x: x[2], reverse=True) 
    top_2_eq, bottom_2_eq = ord_centers_eq[:2], ord_centers_eq[2:]
    top_2_eq.sort(key=lambda x: x[1])
    bottom_2_eq.sort(key=lambda x: x[1])

    cluster_map_eq = {top_2_eq[0][0]: "Grassroot Adopters", top_2_eq[1][0]: "Leapfroggers", bottom_2_eq[0][0]: "Low Demand Economies", bottom_2_eq[1][0]: "Tokenization Hubs"}
    df['Equal_Archetype'] = kmeans_eq.predict(cluster_scaled_eq)
    df['Equal_Archetype'] = df['Equal_Archetype'].map(cluster_map_eq)

    # Visual Helper Columns
    np.random.seed(42) 
    df['regulation_jittered'] = df['regulation'] + np.random.uniform(-0.25, 0.25, size=len(df))
    df['Flag'] = df['ISO Code'].apply(get_flag)
    df['Country_Flag'] = df['Flag'] + " " + df['Country']

    return df.sort_values(by=['Country'])

df = load_data()
dataset_size = len(df)

# ==========================================
# 4. SLIDER INTERPOLATION LOGIC
# ==========================================
slider_options = []
alpha_map = {}
for i in range(0, 101, 5):
    a = i / 100.0
    wc = (1 - a) * 52.5 + a * 33.333
    wi = (1 - a) * 0.7 + a * 33.333
    wa = (1 - a) * 46.8 + a * 33.333
    
    label = f"Closedness: {wc:.1f}% | Crypto Adoption: {wa:.1f}% | Inflation: {wi:.1f}%"
    slider_options.append(label)
    alpha_map[label] = a

# ==========================================
# 5. INITIALIZE STATE & CALLBACKS 
# ==========================================
if 'sim_c' not in st.session_state:
    st.session_state.sim_c = "United States"

if 'model_slider' not in st.session_state:
    st.session_state.model_slider = slider_options[0]
    st.session_state.slider_tab1 = slider_options[0]
    st.session_state.slider_tab2 = slider_options[0]
    st.session_state.slider_tab3 = slider_options[0]

r_init = df[df['Country'] == st.session_state.sim_c].iloc[0]
if 'sim_reg' not in st.session_state: st.session_state.sim_reg = float(r_init['regulation'])
if 'sim_inf' not in st.session_state: st.session_state.sim_inf = float(r_init['Inflation'])
if 'sim_close' not in st.session_state: st.session_state.sim_close = float(r_init['Financial_Closedness'])
if 'sim_adopt' not in st.session_state: st.session_state.sim_adopt = int(r_init['Crypto_Adoption_Rank'])

if 'w_close' not in st.session_state: st.session_state.w_close = 52.5
if 'w_adopt' not in st.session_state: st.session_state.w_adopt = 46.8

if 'pca_models' not in st.session_state:
    pca_features = ['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank']
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(df[pca_features])
        
    scaler_cluster_pca = StandardScaler()
    cluster_scaled_pca = scaler_cluster_pca.fit_transform(df[['regulation', 'PCA_Index_Score']])
    kmeans_pca = KMeans(n_clusters=4, random_state=42, n_init=10).fit(cluster_scaled_pca)
    
    centers_pca = scaler_cluster_pca.inverse_transform(kmeans_pca.cluster_centers_)
    ord_centers_pca = [(i, cx, cy) for i, (cx, cy) in enumerate(centers_pca)]
    ord_centers_pca.sort(key=lambda x: x[2], reverse=True) 
    top_2, bottom_2 = ord_centers_pca[:2], ord_centers_pca[2:]
    top_2.sort(key=lambda x: x[1])
    bottom_2.sort(key=lambda x: x[1])

    cluster_map_pca = {
        top_2[0][0]: "Grassroot Adopters", top_2[1][0]: "Leapfroggers", 
        bottom_2[0][0]: "Low Demand Economies", bottom_2[1][0]: "Tokenization Hubs"
    }

    st.session_state['pca_models'] = {
        'scaler_pca': scaler_pca, 'scaler_cluster': scaler_cluster_pca, 
        'kmeans': kmeans_pca, 'cluster_mapping': cluster_map_pca
    }

def sync_sliders(source_key):
    val = st.session_state[source_key]
    st.session_state.model_slider = val
    st.session_state.slider_tab1 = val
    st.session_state.slider_tab2 = val
    st.session_state.slider_tab3 = val

def handle_country_change():
    c = st.session_state.sim_c
    r = df[df['Country'] == c].iloc[0]
    st.session_state.sim_reg = float(r['regulation'])
    st.session_state.sim_inf = float(r['Inflation'])
    st.session_state.sim_close = float(r['Financial_Closedness'])
    st.session_state.sim_adopt = int(r['Crypto_Adoption_Rank'])

def reset_simulator():
    handle_country_change() 
    st.session_state.comp_countries = [] 
    st.session_state.w_close = 52.5
    st.session_state.w_adopt = 46.8
    
    st.session_state.model_slider = slider_options[0]
    st.session_state.slider_tab1 = slider_options[0]
    st.session_state.slider_tab2 = slider_options[0]
    st.session_state.slider_tab3 = slider_options[0]

def update_weights(changed):
    if changed == 'close':
        if st.session_state.w_close + st.session_state.w_adopt > 100.0:
            st.session_state.w_adopt = 100.0 - st.session_state.w_close
    elif changed == 'adopt':
        if st.session_state.w_close + st.session_state.w_adopt > 100.0:
            st.session_state.w_close = 100.0 - st.session_state.w_adopt

# ==========================================
# 6. CALCULATE ACTIVE INTERPOLATED MODEL
# ==========================================
active_label = st.session_state.model_slider
alpha = alpha_map[active_label]

wc_active = (1 - alpha) * 0.525 + alpha * 0.33333
wi_active = (1 - alpha) * 0.007 + alpha * 0.33333
wa_active = (1 - alpha) * 0.468 + alpha * 0.33333

X_scaled_active = st.session_state['pca_models']['scaler_pca'].transform(df[['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank']])
active_raw = (X_scaled_active[:, 0] * wc_active) + (X_scaled_active[:, 1] * wi_active) - (X_scaled_active[:, 2] * wa_active)

df['Active_Index_Score'] = ((active_raw - active_raw.min()) / (active_raw.max() - active_raw.min())) * 100

scaler_cluster_active = StandardScaler()
cluster_scaled_active = scaler_cluster_active.fit_transform(df[['regulation', 'Active_Index_Score']])
kmeans_active = KMeans(n_clusters=4, random_state=42, n_init=10).fit(cluster_scaled_active)

centers_act = scaler_cluster_active.inverse_transform(kmeans_active.cluster_centers_)
ord_centers_act = [(i, cx, cy) for i, (cx, cy) in enumerate(centers_act)]
ord_centers_act.sort(key=lambda x: x[2], reverse=True) 
top_2_act, bottom_2_act = ord_centers_act[:2], ord_centers_act[2:]
top_2_act.sort(key=lambda x: x[1])
bottom_2_act.sort(key=lambda x: x[1])

cluster_map_active = {top_2_act[0][0]: "Grassroot Adopters", top_2_act[1][0]: "Leapfroggers", bottom_2_act[0][0]: "Low Demand Economies", bottom_2_act[1][0]: "Tokenization Hubs"}
df['Active_Archetype'] = kmeans_active.predict(cluster_scaled_active)
df['Active_Archetype'] = df['Active_Archetype'].map(cluster_map_active)

# Helper function to generate the permanent label block under sliders
def get_permanent_label():
    clean_label = st.session_state.model_slider.replace('PCA Weights ← ', '').replace(' → Equal Weights', '')
    return f"<div style='text-align: center; font-size: 1.15rem; font-weight: 800; color: #2C3E50; margin-top: 10px; padding: 10px; background-color: #FFFFFF; border-radius: 5px; border: 1px solid #E0E5EC;'>Use the slider to change how the factors are balanced. Move the slider to see the factors go from PCA-based to equally weighted</div>"

# ==========================================
# APP HEADER
# ==========================================
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("https://images.ctfassets.net/sygt3q11s4a9/3x7SlEtglsK24xKCI4klI5/ce347e6caf775dd7d8a7759619577871/1_oOgJJrP9DcjOLpq5YLzsFQ.png?fm=avif&w=1400&h=712&q=65", width=200)
with col_text:
    st.markdown("<h2 style='color: #2C3E50; margin-top: -5px; font-size: 1.8rem;'>From the Unbanked to the Unbrokered</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #0052FF; margin-bottom: 0px;'>Tokenized Capital Markets Propensity Insights</h1>", unsafe_allow_html=True)

st.markdown("<p style='font-size: 1.1rem; color: #555; margin-bottom: 20px;'>Insights into the likely adoption drivers of tokenized capital markets.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Visual Dashboard", 
    "📊 Raw Data Explorer", 
    "🧩 Propensity Archetypes",
    "🎛️ The \"What-If\" Simulator",
    "📋 Methodology"
])

# ==========================================
# TAB 1: VISUAL DASHBOARD
# ==========================================
with tab1:
    st.markdown(get_permanent_label(), unsafe_allow_html=True)
    st.select_slider(
        "** **", 
        options=slider_options,
        key='slider_tab1',
        on_change=sync_sliders,
        args=('slider_tab1',)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    fig_map = px.choropleth(
        df, locations="ISO Code", color="Active_Archetype", hover_name="Country_Flag",
        labels={
            'Active_Index_Score': 'SoV Index Score', 
            'regulation': 'Regulation', 
            'Inflation': 'Inflation (%)', 
            'Financial_Closedness': 'Financial Closedness', 
            'Crypto_Adoption_Rank': 'Crypto Adoption Rank',
            'Active_Archetype': 'Archetype'
        },
        hover_data={
            "ISO Code": False, 
            'Active_Archetype': True,
            'Active_Index_Score': ':.1f', 
            "regulation": ':.1f', 
            "Inflation": ':.1f', 
            "Financial_Closedness": ':.2f', 
            "Crypto_Adoption_Rank": True
        },
        color_discrete_map=color_map,
        projection="natural earth",
        title="<b>Propensity Archetypes</b>"
    )
    
    fig_map.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.divider()

    y_countries = sorted(df['Country'].unique())
    col_sel_a, col_sel_b = st.columns(2)
    
    with col_sel_a:
        c1 = st.selectbox("📍 Select Country A", y_countries, index=y_countries.index("United States") if "United States" in y_countries else 0, key="t1_c1")
        r1 = df[df['Country'] == c1].iloc[0]
        st.markdown(f"### {r1['Flag']} {c1} Snapshot")
        st.metric("Propensity Archetype", f"{r1['Active_Archetype']}")
        m2, m3, m4, m5 = st.columns(4)
        m2.metric("Store of Value Index Score", f"{r1['Active_Index_Score']:.1f}")
        m3.metric("Regulation", f"{r1['regulation']:.1f}")
        m4.metric("Crypto Adoption", f"#{int(r1['Crypto_Adoption_Rank'])}")
        m5.metric("Inflation", f"{r1['Inflation']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_sel_b:
        c2 = st.selectbox("📍 Select Country B (Comparison)", ["(None)"] + y_countries, index=0, key="t1_c2")
        if c2 != "(None)":
            r2 = df[df['Country'] == c2].iloc[0]
            st.markdown(f"### {r2['Flag']} {c2} Snapshot")
            st.metric("Propensity Archetype", f"{r2['Active_Archetype']}")
            m2b, m3b, m4b, m5b = st.columns(4)
            m2b.metric("Store of Value Index Score", f"{r2['Active_Index_Score']:.1f}")
            m3b.metric("Regulation", f"{r2['regulation']:.1f}")
            m4b.metric("Crypto Adoption", f"#{int(r2['Crypto_Adoption_Rank'])}")
            m5b.metric("Inflation", f"{r2['Inflation']:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Select a second country from the dropdown above to compare structural metrics side-by-side.")

# ==========================================
# TAB 2: RAW DATA EXPLORER
# ==========================================
with tab2:
    st.markdown(get_permanent_label(), unsafe_allow_html=True)
    st.select_slider(
        "** **", 
        options=slider_options,
        key='slider_tab2',
        on_change=sync_sliders,
        args=('slider_tab2',)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    display_cols = ['Country_Flag', 'Active_Archetype', 'Active_Index_Score', 'PCA_Index_Score', 'Equal_Index_Score', 'regulation', 'Crypto_Adoption_Rank', 'Inflation', 'Financial_Closedness']
    df_tab2 = df[display_cols].set_index('Country_Flag').sort_values(by='Active_Index_Score', ascending=False)
    
    st.subheader("Archetypes and Variables")
    st.caption("Displaying the active custom variables next to the fixed baseline models for direct comparison.")
    
    def style_rows_by_archetype(row):
        arch = row['Active_Archetype']  
        if arch == 'Grassroot Adopters': color = '#FDEAEA' 
        elif arch == 'Leapfroggers': color = '#EAF8F1' 
        elif arch == 'Low Demand Economies': color = '#F4EDF7' 
        elif arch == 'Tokenization Hubs': color = '#EAF3FB' 
        else: color = '#FFFFFF' 
        return [f'background-color: {color}; color: #2C3E50'] * len(row)

    styled_df = df_tab2.style.apply(style_rows_by_archetype, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=500, column_config={
        "Country_Flag": st.column_config.TextColumn(
            "Country",
            help="The country name and its ISO flag."
        ),
        "Active_Archetype": st.column_config.TextColumn(
            "🏛️ Active Archetype",
            help="The K-Means clustering classification (Grassroot Adopters, Leapfroggers, Tokenization Hubs, or Low Demand Economies) dynamically based on the current slider position."
        ),
        "Active_Index_Score": st.column_config.ProgressColumn(
            "🎯 Active Score", 
            min_value=0, max_value=100, format="%.1f",
            help="The Store of Value Necessity Index score calculated dynamically based on the current slider weights."
        ),
        "PCA_Index_Score": st.column_config.NumberColumn(
            "📊 Base PCA Score", format="%.1f",
            help="The static baseline score calculated strictly using the Principal Component Analysis (PCA) weights."
        ),
        "Equal_Index_Score": st.column_config.NumberColumn(
            "🟰 Equal Weights", format="%.1f",
            help="The static baseline score calculated forcing a strictly equal 33.3% weight across all variables."
        ),
        "regulation": st.column_config.NumberColumn(
            "⚖️ Regulation (0-8)", format="%.1f",
            help="Tracks the maturity and legality of a nation's formal digital asset frameworks (0 = Lowest, 8 = Highest)."
        ),
        "Crypto_Adoption_Rank": st.column_config.NumberColumn(
            "🏆 Crypto Adoption Rank",
            help="Real-world utility and grassroots adoption of digital assets by everyday retail users. Rank 1 = Highest Global Crypto Adoption."
        ),
        "Inflation": st.column_config.NumberColumn(
            "💸 Inflation (%)", format="%.1f%%",
            help="The annual percentage change in the cost of domestically manufactured goods and services."
        ),
        "Financial_Closedness": st.column_config.NumberColumn(
            "🏦 Closedness Score", format="%.2f",
            help="Measures capital controls and restrictions on cross-border financial transactions. A higher positive number indicates stricter controls and a more closed economy."
        )
    })

# ==========================================
# TAB 3: PROPENSITY ARCHETYPES (INTERACTIVE)
# ==========================================
with tab3:
    st.markdown(get_permanent_label(), unsafe_allow_html=True)
    st.select_slider(
        "** **", 
        options=slider_options,
        key='slider_tab3',
        on_change=sync_sliders,
        args=('slider_tab3',)
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.header("Propensity Archetypes in Clusters")
    st.markdown("By mapping SoV index against our Regulatory Score, countries fall into four different categories.")

    fig_quad = px.scatter(
        df, x='regulation_jittered', y='Active_Index_Score', color='Active_Archetype',
        color_discrete_map=color_map, hover_name='Country_Flag',
        text='ISO Code',
        labels={'Active_Index_Score': 'SoV Index Score', 'regulation_jittered': 'Regulation', 'Inflation': 'Inflation (%)', 'Financial_Closedness': 'Financial Closedness', 'Crypto_Adoption_Rank': 'Crypto Adoption Rank', 'Active_Archetype': 'Archetype'},
        hover_data={'regulation_jittered': False, 'Active_Archetype': True, 'Active_Index_Score': ':.1f', 'regulation': ':.1f', 'Inflation': ':.1f', 'Financial_Closedness': ':.2f', 'Crypto_Adoption_Rank': True},
        size_max=15
    )

    fig_quad.update_traces(
        textposition='top center',
        textfont=dict(size=10, color='#2C3E50'),
        marker=dict(size=14, line=dict(width=1, color='black')), 
        opacity=0.85
    )

    fig_quad.add_annotation(x=2.0, y=102, text="<b>Grassroot Adopters</b>", showarrow=False, font=dict(color="#e74c3c", size=15))
    fig_quad.add_annotation(x=6.0, y=102, text="<b>Leapfroggers</b>", showarrow=False, font=dict(color="#2ecc71", size=15))
    fig_quad.add_annotation(x=2.0, y=-2, text="<b>Low Demand Economies</b>", showarrow=False, font=dict(color="#9b59b6", size=15))
    fig_quad.add_annotation(x=6.0, y=-2, text="<b>Tokenization Hubs</b>", showarrow=False, font=dict(color="#3498db", size=15))

    fig_quad.update_layout(
        xaxis_title="<b>Formal Regulatory Framework Score (0-8)</b>",
        yaxis_title="<b>Store of Value Necessity Index (Active Model)</b>",
        xaxis=dict(range=[-0.5, 8.5], tickmode='linear', tick0=0, dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
        plot_bgcolor="white", height=700, legend_title_text="Propensity Archetypes", margin=dict(t=30, b=30, l=30, r=30)
    )

    st.plotly_chart(fig_quad, use_container_width=True)

    st.divider()
    
    st.subheader("Archetype Breakdown")
    
    # --- ROW 1 ---
    r1_c1, r1_c2 = st.columns(2)
    
    with r1_c1:
        st.markdown("### 🔴 Grassroot Adopters")
        st.markdown("""
        **Demand / High-Constraint, State-Led:** Jurisdictions where macroeconomic pressures—such as sustained inflation, currency instability, or capital controls—create strong demand for alternative financial infrastructure and digital assets. But adoption is typically managed within governmental frameworks rather than through open market development. In these settings, authorities may recognize the functional utility of blockchain-based technologies, particularly for payments, settlement, or reducing reliance on external financial networks, while favoring centralized, state-supervised implementations over permissionless models.
        """)
        
    with r1_c2:
        st.markdown("### 🟢 Leapfroggers")
        st.markdown("""
        **Adaptive, Market-Driven:** Jurisdictions with relatively strong regulatory clarity for digital assets combined with high inflation, currency instability, or restrictions on cross-border transactions. In these settings, tokenized assets and blockchain-based financial infrastructure often develop not solely as investment vehicles, but as practical mechanisms for value preservation, liquidity access, and transaction efficiency. Adoption may be driven by households, firms, and financial intermediaries seeking alternatives to legacy systems that are either unstable, costly, or limited in their ability to support international flows.
        """)

    st.markdown("<br>", unsafe_allow_html=True) 

    # --- ROW 2 ---
    r2_c1, r2_c2 = st.columns(2)
    
    with r2_c1:
        st.markdown("### 🟣 Low Demand Economies")
        st.markdown("""
        **Market Indifference:** These regions sit at the intersection of low macroeconomic distress and low regulatory clarity. With relatively stable local currencies and accessible traditional banking, everyday citizens lack the acute "survival" catalyst needed to organically adopt unbrokered digital assets. Because the grassroots demand is low, local governments have little incentive or urgency to proactively draft comprehensive digital asset frameworks.
        """)

    with r2_c2:
        st.markdown("### 🔵 Tokenization Hubs")
        st.markdown("""
        **Optimization + Institutional Arbitrage:** These are wealthy, stable, financial hubs. Because inflation is low and capital mobility is high, retail demand for "life raft" crypto is negligible. Instead, the push for tokenization in these jurisdictions is entirely institutional. Governments here are enacting regulations designed to lure global capital and traditional finance (TradFi) institutions seeking efficiency gains, operational optimization, and jurisdictional arbitrage.
        """)

# ==========================================
# TAB 4: POLICY SIMULATOR (WHAT-IF)
# ==========================================
with tab4:
    st.header("The \"What-If\" Simulator")
    st.markdown("""
    Our baseline weights were determined by **Principal Component Analysis (PCA)**. PCA is a machine learning algorithm that mathematically discovers the true variance within the global economy to automatically weight each variable without human bias. To see the effects of changes in these inputs, select a country and dynamically shift its core parameters—or override the global PCA weights—to see how these changes mathematically redefine that country within the global landscape.
    """)
    st.divider()
    
    col_shock, col_weights = st.columns(2)
    
    with col_shock:
        st.subheader("1. Shock the System")
        
        c_sel, c_res = st.columns([3, 1])
        with c_sel:
            sim_country = st.selectbox("Primary Country to Simulate", sorted(df['Country'].unique()), key="sim_c", on_change=handle_country_change)
        with c_res:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            st.button("🔄 Reset", on_click=reset_simulator, help="Reset all sliders and weights to their default values for this country.", use_container_width=True)
            
        r_sim = df[df['Country'] == sim_country].iloc[0]
        st.caption(f"Adjust the specific metrics for {sim_country}.")
        
        sim_reg = st.slider("Formal Regulation Framework (0-8)", 0.0, 8.0, step=0.5, key="sim_reg",
                            help="Tracks the maturity, legality, and comprehensiveness of a nation's formal digital asset frameworks (Atlantic Council).")
        
        sim_inf = st.slider("Inflation (%)", -5.0, 150.0, step=1.0, key="sim_inf",
                            help="The annual percentage change in the cost of domestically manufactured goods and services via the GDP deflator (World Bank).")
        
        sim_close = st.slider("Financial Closedness Score (-2.5 to 2.5)", -2.5, 2.5, step=0.1, key="sim_close",
                              help="Adapted from the Chinn-Ito Index. -2.5 represents fully open capital markets, while 2.5 represents strictly closed economies closed economies.")
        
        sim_adopt = st.slider("Crypto Adoption Rank (1 = Highest)", 1, 150, step=1, key="sim_adopt",
                              help="Real-world utility and adoption of digital assets by everyday retail users (Chainalysis). Rank 1 = Highest Adoption.")
        
    with col_weights:
        st.subheader("2. Compare & Override")
        comp_countries = st.multiselect(
            "Select Benchmark Countries to Overlay", 
            sorted(df['Country'].unique()), 
            key="comp_countries",
            help="Select one or more countries to display as fixed reference points on the graph."
        )
        
        st.caption("Shift the PCA variance weights. The system caps combinations to strictly equal 100%.")
        
        st.slider("Weight: Financial Closeness (%)", 0.0, 100.0, step=0.1, key='w_close', on_change=update_weights, args=('close',),
                  help="Override the PCA variance weight for Financial Closedness.")
        
        st.slider("Weight: Crypto Adoption Score(%)", 0.0, 100.0, step=0.1, key='w_adopt', on_change=update_weights, args=('adopt',),
                  help="Override the PCA variance weight for the Crypto Adoption Score.")
        
        w_close = st.session_state.w_close
        w_adopt = st.session_state.w_adopt
        w_inf = round(100.0 - w_close - w_adopt, 1)
        
        st.metric(label="Weight: Inflation (%)", value=f"{w_inf:.1f}%", 
                  help="Auto-calculated remainder to ensure a perfect 100% distribution. Represents the override weight for Inflation.")

    wc_n, wa_n, wi_n = w_close/100.0, w_adopt/100.0, w_inf/100.0
    models = st.session_state['pca_models']
    
    all_features = df[['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank']]
    all_scaled = models['scaler_pca'].transform(all_features)
    
    all_raw_scores = (all_scaled[:, 0] * wc_n) + (all_scaled[:, 1] * wi_n) - (all_scaled[:, 2] * wa_n)
    new_min, new_max = all_raw_scores.min(), all_raw_scores.max()

    new_features_df = pd.DataFrame([[st.session_state.sim_close, st.session_state.sim_inf, st.session_state.sim_adopt]], columns=['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank'])
    new_scaled = models['scaler_pca'].transform(new_features_df)[0]
    
    new_pca_raw = (new_scaled[0] * wc_n) + (new_scaled[1] * wi_n) - (new_scaled[2] * wa_n)
    
    if new_max == new_min: new_index = 50.0
    else: new_index = ((new_pca_raw - new_min) / (new_max - new_min)) * 100
        
    new_index = np.clip(new_index, 0, 100)
    
    new_point_df = pd.DataFrame({'regulation': [st.session_state.sim_reg], 'PCA_Index_Score': [new_index]})
    new_cluster_scaled = models['scaler_cluster'].transform(new_point_df)
    predicted_cluster = models['kmeans'].predict(new_cluster_scaled)[0]
    new_arch = models['cluster_mapping'][predicted_cluster]

    st.divider()
    st.subheader(f"Projected Impact for {sim_country}")
    
    col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
    col_res1.metric("New Simulated SoV Index Score", f"{new_index:.1f}", f"{new_index - r_sim['PCA_Index_Score']:.1f} vs. original PCA")
    col_res2.metric("New Archetype Classification", new_arch)

    df_sim_bg = df.copy()
    df_sim_bg['Sim_Raw'] = all_raw_scores
    if new_max == new_min: df_sim_bg['Sim_Index'] = 50.0
    else: df_sim_bg['Sim_Index'] = ((df_sim_bg['Sim_Raw'] - new_min) / (new_max - new_min)) * 100
        
    bg_features = df_sim_bg[['regulation', 'Sim_Index']].rename(columns={'Sim_Index': 'PCA_Index_Score'})
    bg_cluster_scaled = models['scaler_cluster'].transform(bg_features)
    bg_clusters = models['kmeans'].predict(bg_cluster_scaled)
    df_sim_bg['Sim_Archetype'] = [models['cluster_mapping'][c] for c in bg_clusters]

    fig_sim = px.scatter(
        df_sim_bg, x='regulation', y='Sim_Index', color='Sim_Archetype',
        color_discrete_map=color_map, hover_name='Country_Flag', opacity=0.2,
        size_max=10
    )
    
    for c in comp_countries:
        if c != sim_country: 
            r_comp = df_sim_bg[df_sim_bg['Country'] == c].iloc[0]
            fig_sim.add_trace(go.Scatter(
                x=[r_comp['regulation']], y=[r_comp['Sim_Index']],
                mode='markers+text', text=[c], textposition='top center', textfont=dict(color="black", size=10),
                marker=dict(size=12, color='white', line=dict(width=2, color='gray')),
                name=f"Benchmark: {c}", hoverinfo="none"
            ))

    fig_sim.add_trace(go.Scatter(
        x=[r_sim['regulation']], y=[r_sim['PCA_Index_Score']],
        mode='markers', marker=dict(size=14, color='white', line=dict(width=2, color='black')),
        name=f"Original {sim_country}", hoverinfo="none"
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=[st.session_state.sim_reg], y=[new_index],
        mode='markers', marker=dict(size=22, symbol='star', color=color_map.get(new_arch, '#000'), line=dict(width=2, color='black')),
        name=f"Simulated {sim_country}", hoverinfo="none"
    ))
    
    fig_sim.add_annotation(
        x=st.session_state.sim_reg, y=new_index, ax=r_sim['regulation'], ay=r_sim['PCA_Index_Score'],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black"
    )
    
    fig_sim.update_layout(
        xaxis_title="<b>Regulation Score</b>", yaxis_title="<b>Custom Index Score</b>",
        xaxis=dict(range=[-0.5, 8.5]), yaxis=dict(range=[-5, 105]),
        plot_bgcolor="white", height=600, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================
# TAB 5: DATA AND METHODOLOGY
# ==========================================
with tab5:
    st.header("Data Handling and Methodology")
    st.markdown("This section describes how the data in our pipeline is assembled, how missing values are treated at each step, how variables are standardized, and how our dimensionality reduction (PCA) and K-Means clustering models are specified.")
    
    st.markdown("### 1. Import and Merge")
    st.markdown(f"The analysis starts from four raw data sources and our own assessments to ensure a comprehensive macroeconomic snapshot of {dataset_size} countries:")
    
    st.markdown("""
    * **The Chinn-Ito index (KAOPEN):** An index measuring a country’s degree of capital account openness. The index was initially introduced in Chinn and Ito (Journal of Development Economics, 2006). KAOPEN is based on the binary dummy variables that codify the tabulation of restrictions on cross-border financial transactions reported in the IMF’s Annual Report on Exchange Arrangements and Exchange Restrictions (AREAER). This update is based on AREAER 2024, which contains the information on regulatory restrictions on cross-border financial transactions as of the end of 2023.More information on its construction can be found in: <br><br>“[A New Measure of Financial Openness](http://web.pdx.edu/~ito/kaopen_Chinn-Ito_hi0523.pdf)”, Journal of Comparative Policy Analysis, Volume 10, Issue 3 September 2008, p. 309 - 322. Note that this paper uses the 2007 version of the dataset (containing data up to only 2005), which will differ from the current version of the dataset. The full and current dataset that we used is available at [Chinn-Ito website](http://web.pdx.edu/~ito/Chinn-Ito_website.htm).<br><br>
    * **Regulatory Frameworks (Atlantic Council data + Coinbase Institute assessment):** Tracks the maturity and legality of formal digital asset frameworks on a continuous scale of 0 to 8. The 0 to 8 scale has been developed by looking at the Atlantic Council’s assessment of a country's legal framework—based on the legal status of cryptocurrencies, their taxation, AML/CFT, Consumer Protection, and Licensing—as well as our in-house regulatory team’s assessment of a country's stablecoin regulations. More information on the Atlantic Council dataset can be found in: <br><br>Atlantic Council, "[Cryptocurrency Regulation Tracker](https://www.atlanticcouncil.org/programs/geoeconomics-center/cryptoregulationtracker/)," GeoEconomics Center, July 2025.<br><br>
    * **Crypto Adoption (Chainalysis):** Contains a rank measure of real-world cryptocurrency adoption by country (1 = highest adoption). The Chainalysis report holds the Global Crypto Adoption Index—a metric designed to identify where “grassroots” adoption is strongest by weighting on-chain activity against population size and purchasing power parity (PPP). For our ranking, we used the “Overall index ranking” variable. The full report and dataset can be found in:<br><br> Chainalysis. [The 2023 Geography of Cryptocurrency Report](https://www.chainalysis.com/blog/2023-global-crypto-adoption-index/). New York: Chainalysis, October 2023.<br><br>
    * **World Bank WDI:** Provides macroeconomic indicators, specifically the annual Inflation percentage (via the GDP deflator). Lastly, we use the inflation indicator from the World Bank’s World Development Indicators. The full dataset can be found in: <br><br>World Bank. "[Inflation, consumer prices (annual %).](https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG)" World Development Indicators. Accessed April 10, 2026.
    """, unsafe_allow_html=True)
    
    st.markdown("The harmonized dataframes are merged via outer joins on `['ISO Code', 'Year']`. Every country present in any source appears in the merged dataframe. If a given source has no observation for that country, its columns contribute `NaN` for that row. At this stage, nothing is dropped; missing values are allowed to exist and are managed downstream.")

    st.divider()

    st.markdown("### 2. Data Cleaning and NA Treatment")
    st.markdown("""
    After merging, we ensure that every row corresponds to a valid country within the intended sample window:
    * **Time Restriction:** We retain only observations with `Year = 2023`. All earlier years are removed.
    * **Converting to Numeric:** Raw data sources often represent missing values as strings (e.g., `".."`). The script replaces these literal strings with `NaN` and applies `pd.to_numeric(..., errors='coerce')`. This standardizes all non-ID variables as numeric and uses `NaN` as the single missing value marker.
    * **Retaining Outliers:** Hyper-outliers (such as Venezuela and Zimbabwe) are intentionally kept in the dataset. No Winsorization or data clipping is applied, ensuring the model fully accounts for the mathematical severity of global macroeconomic extremes.
    * **Core NA Filter:** Finally, we perform a minimal essential NA filter. We drop any row where one of the core modeling variables is missing: `Financial_Closedness`, `Inflation`, `Crypto_Adoption_Rank`, or `regulation`. This guarantees that every row in the cleaned panel has complete data for the primary spatial mapping.
    """)

    st.divider()

    st.markdown("### 3. Standardization and Dimensionality Reduction (PCA)")
    st.markdown("""
    Because our variables are measured in completely different units (ranks, percentages, and indices), we must standardize them before combining them into a single metric.
    * Using scikit-learn’s `StandardScaler`, we compute the mean and standard deviation across the cleaned sample for the three core variables: `Financial_Closedness`, `Inflation`, and `Crypto_Adoption_Rank`. This rescales each variable so that it has a mean of 0 and a variance of 1.
    * Instead of manually guessing which factor drives adoption, we utilize **Principal Component Analysis (PCA)**. The PCA algorithm discovers the true variance within the global economy and automatically assigns weights to our standardized variables. The baseline PCA model applies weights of **52.5% to Closedness**, **46.8% to Adoption**, and **0.7% to Inflation**. A more in depth explanation of the Principal Component Analysis can be found in: [Greenacre, M., Groenen, P.J.F., Hastie, T. et al. Principal component analysis. Nat Rev Methods Primers 2, 100 (2022)](https://doi.org/10.1038/s43586-022-00184-w).
    * We also calculate an **Equal Weights Model**, forcing a strict 33.3% weight across all three variables to isolate and highlight the impact of extreme inflation.
    * The raw scores from both models are then min-max normalized onto a clean **0 to 100 Store of Value (SoV) Index Score**, where 100 represents the highest macroeconomic necessity for digital assets.
    """)
    
    st.divider()

    st.markdown("### 4. K-Means Clustering Specification (Propensity Archetypes)")
    st.markdown("""
    To objectively classify countries into our four geopolitical archetypes without relying on arbitrary human threshold lines, we utilize unsupervised **K-Means Clustering**:
    1. We isolate the two mapping axes: The `regulation` score (X-Axis) and the `Index_Score` (Y-Axis).
    2. We apply `StandardScaler` to both axes. This is critical: it ensures that the larger 0-100 scale of the Y-Axis does not mathematically overpower the smaller 0-8 scale of the X-Axis during distance calculations.
    3. We instruct the algorithm to find exactly four clusters (`n_clusters=4`). The algorithm measures the spatial distance between every country and organically groups them based on mathematical density.
    4. The coordinates of the resulting four centroids are mathematically sorted into quadrants (Top-Left, Top-Right, Bottom-Left, Bottom-Right) and assigned to their respective behavioral labels: **Grassroot Adopters**, **Leapfroggers**, **Low Demand Economies**, and **Tokenization Hubs**. A more in depth explanation can be found in: [Chris Piech, "K-Means Clustering Algorithm," CS221: Artificial Intelligence: Principles and Techniques, Stanford University](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html).
    """)
    
# ==========================================
# GLOBAL FOOTER (Displays on all tabs)
# ==========================================
st.markdown("<div class='footer'>", unsafe_allow_html=True)
col_foot1, col_foot2 = st.columns(2)

with col_foot1:
    st.markdown("#### 📚 Variable Definitions")
    st.markdown("""
    * **Store of Value (SoV) Index Score:** A composite 0-100 score assessing a population's macro-necessity for a store of value this is made up from the below variables. By default, the Crypto Adoption Rank, Inflation data, and Financial Closedness value are weighted using Principal Component Analysis (PCA), which uses patterns in the data to determine the weight of each variable. Users can also adjust these weights using the slider or the “What if” tab.
    * **Regulatory Frameworks:** Tracks the maturity and legality of formal digital asset frameworks on a continuous scale of 0 to 8 (Atlantic Council data + Coinbase Institute assessment).
    * **Crypto Adoption Rank:** Real-world utility and adoption of digital assets by everyday retail users (Chainalysis). *Rank 1 = Highest Crypto Adoption.*
    * **Inflation (%):** The annual percentage change in the cost of domestically manufactured goods and services (World Bank).
    * **Financial Closedness:** Measures capital controls and restrictions on cross-border financial transactions (Chinn-Ito Index, inverted).
    """)

with col_foot2:
    st.markdown("#### ⚙️ Quick Methodology Summary")
    st.markdown("""
    * **Data Pipeline:** Aggregates macro data from 6 leading global institutions. Hyper-outliers (e.g. VEN, ZWE) are intentionally retained to accurately reflect global macroeconomic extremes without clipping.
    * **SoV Index:** Baseline weights are mathematically derived using **Principal Component Analysis (PCA)** to prioritize systemic closedness, actively removing human guessing.
    * **Machine Learning Archetypes:** Unsupervised Machine Learning Archetypes: We use a pattern-recognition algorithm that groups countries into four natural clusters based on shared similarities and patters they are.
    """)
    st.markdown(f"**Total Countries Analyzed:** {dataset_size}")

st.markdown("</div>", unsafe_allow_html=True)
