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
    page_title="The Tokenized Equities Propensity Index", 
    page_icon="©️", 
    layout="wide"
)

st.markdown("""
    <style>
        h1, h2, h3 { font-family: 'Inter', sans-serif; }
        [data-testid="stMetricValue"] { color: #0052FF !important; font-weight: 800; font-size: 1.8rem;}
        .stTabs [aria-selected="true"] {
            background-color: #F0F4F8;
            border-bottom: 3px solid #0052FF !important;
            color: #0052FF !important;
            font-weight: bold;
        }
        .snapshot-box {
            background-color: #F8F9FA;
            border: 1px solid #E0E5EC;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER DICTIONARIES (FLAGS)
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
    "Sovereign Controllers": "#e74c3c",   
    "Leapfroggers": "#2ecc71",            
    "Low Demand Economies": "#9b59b6",    
    "Financial Hubs": "#3498db"                   
}

# ==========================================
# 3. DATA LOADING PIPELINE (Safely Cached)
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "df_unbrokered_master.csv")
    df = pd.read_csv(csv_path)
    
    col_mapping = {'Archetype_Label': 'Archetype'}
    for col in df.columns:
        if 'Financial_Closedness' in col: col_mapping[col] = 'Financial_Closedness'
        elif 'Inflation' in col: col_mapping[col] = 'Inflation'
        elif 'Crypto_Adoption_Rank' in col: col_mapping[col] = 'Crypto_Adoption_Rank'
            
    df = df.rename(columns=col_mapping)
    df['Archetype'] = df['Archetype'].replace({'Low Demand': 'Low Demand Economies', 'Giants': 'Financial Hubs'})
    
    c_min, c_max = df['Financial_Closedness'].min(), df['Financial_Closedness'].max()
    df['Financial_Closedness_Display'] = ((df['Financial_Closedness'] - c_min) / (c_max - c_min)) * 100
    
    np.random.seed(42) 
    df['regulation_jittered'] = df['regulation'] + np.random.uniform(-0.25, 0.25, size=len(df))
    
    df['Flag'] = df['ISO Code'].apply(get_flag)
    df['Country_Flag'] = df['Flag'] + " " + df['Country']
    
    return df.sort_values(by=['Country'])

df = load_data()

# ==========================================
# 3.5. MACHINE LEARNING MODELS (Uncached Session State)
# ==========================================
if 'pca_models' not in st.session_state:
    # 1. Build PCA
    pca_features = ['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank']
    scaler_pca = StandardScaler()
    X_scaled = scaler_pca.fit_transform(df[pca_features])
    pca = PCA(n_components=1)
    pc1_scores = pca.fit_transform(X_scaled)
    loadings = pca.components_[0]
    
    if loadings[pca_features.index('Financial_Closedness')] < 0:
        pc1_scores *= -1
    pca_min, pca_max = pc1_scores.min(), pc1_scores.max()

    # 2. Build K-Means
    cluster_data = df[['regulation', 'Index_Score']]
    scaler_cluster = StandardScaler()
    cluster_scaled = scaler_cluster.fit_transform(cluster_data)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(cluster_scaled) 
    
    centers = scaler_cluster.inverse_transform(kmeans.cluster_centers_)

    cluster_mapping = {}
    for i, (reg, idx) in enumerate(centers):
        if reg > 4.0 and idx > 50:
            cluster_mapping[i] = "Leapfroggers"
        elif reg <= 4.0 and idx > 50:
            cluster_mapping[i] = "Sovereign Controllers"
        elif reg <= 4.0 and idx <= 50:
            cluster_mapping[i] = "Low Demand Economies"
        else:
            cluster_mapping[i] = "Financial Hubs"

    st.session_state['pca_models'] = {
        'scaler_pca': scaler_pca, 'pca': pca, 'pca_min': pca_min, 'pca_max': pca_max,
        'scaler_cluster': scaler_cluster, 'kmeans': kmeans, 'cluster_mapping': cluster_mapping
    }

# Sync the df Archetypes strictly with the models we just built
df['Archetype'] = st.session_state['pca_models']['kmeans'].predict(
    st.session_state['pca_models']['scaler_cluster'].transform(df[['regulation', 'Index_Score']])
)
df['Archetype'] = df['Archetype'].map(st.session_state['pca_models']['cluster_mapping'])

if 'sel_country' not in st.session_state:
    st.session_state.sel_country = "United States"

# ==========================================
# 4. APP HEADER
# ==========================================
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("https://images.ctfassets.net/sygt3q11s4a9/3x7SlEtglsK24xKCI4klI5/ce347e6caf775dd7d8a7759619577871/1_oOgJJrP9DcjOLpq5YLzsFQ.png?fm=avif&w=1400&h=712&q=65", width=200)
with col_text:
    st.markdown("<h1 style='color: #0052FF; margin-bottom: 0px;'>From the Unbanked to the Unbrokered</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.3rem; color: #2C3E50; margin-top: 5px;'>Tokenized Capital Markets Propensity Insights—Insights into the likely adoption drivers of tokenized capital markets.</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Visual Dashboard", 
    "📊 Raw Data Explorer", 
    "🧩 Macro Archetypes",
    "🎛️ The \"What-If\" Simulator",
    "📋 Methodology"
])

# ==========================================
# TAB 1: VISUAL DASHBOARD
# ==========================================
with tab1:
    fig_map = px.choropleth(
        df, locations="ISO Code", color="Archetype", hover_name="Country_Flag",
        labels={
            'Index_Score': 'SoV Index Score', 
            'regulation': 'Regulation', 
            'Inflation': 'Inflation (%)', 
            'Financial_Closedness': 'Financial Closedness', 
            'Crypto_Adoption_Rank': 'Adoption Rank',
            'Archetype': 'Archetype'
        },
        hover_data={
            "ISO Code": False, 
            "Archetype": True,
            "Index_Score": ':.1f', 
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
    st.session_state.sel_country = st.selectbox(
        "📍 Select Country for Deep Dive Snapshot", y_countries, 
        index=y_countries.index(st.session_state.sel_country) if st.session_state.sel_country in y_countries else 0, key="t1_c"
    )

    c_row = df[df['Country'] == st.session_state.sel_country]
    if not c_row.empty:
        r = c_row.iloc[0]
        st.markdown(f"### {r['Flag']} {st.session_state.sel_country} Snapshot")
        
        st.markdown("<div class='snapshot-box'>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Macro Archetype", f"{r['Archetype']}")
        m2.metric("SoV Index Score", f"{r['Index_Score']:.1f} / 100")
        m3.metric("Regulation Score", f"{r['regulation']:.1f} / 8")
        m4.metric("Crypto Adoption", f"#{int(r['Crypto_Adoption_Rank'])}")
        m5.metric("Inflation", f"{r['Inflation']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: RAW DATA EXPLORER
# ==========================================
with tab2:
    display_cols = ['Country_Flag', 'Archetype', 'Index_Score', 'regulation', 'Crypto_Adoption_Rank', 'Inflation', 'Financial_Closedness']
    df_tab2 = df[display_cols].set_index('Country_Flag').sort_values(by='Index_Score', ascending=False)
    
    st.subheader("Archetypes and Variables")
    st.caption("Displaying the Core variables and algorithmic Archetype classifications.")
    
    def style_rows_by_archetype(row):
        arch = row['Archetype']
        if arch == 'Sovereign Controllers': color = '#FDEAEA' 
        elif arch == 'Leapfroggers': color = '#EAF8F1' 
        elif arch == 'Low Demand Economies': color = '#F4EDF7' 
        elif arch == 'Financial Hubs': color = '#EAF3FB' 
        else: color = '#FFFFFF' 
        return [f'background-color: {color}; color: #2C3E50'] * len(row)

    styled_df = df_tab2.style.apply(style_rows_by_archetype, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=500, column_config={
        "Archetype": st.column_config.TextColumn("🏛️ Archetype"),
        "Index_Score": st.column_config.ProgressColumn("📊 SoV Index Score", min_value=0, max_value=100, format="%.1f"),
        "regulation": st.column_config.NumberColumn("⚖️ Regulation (0-8)", format="%.1f"),
        "Crypto_Adoption_Rank": st.column_config.NumberColumn("🏆 Adoption Rank"),
        "Inflation": st.column_config.NumberColumn("💸 Inflation (%)", format="%.1f%%"),
        "Financial_Closedness": st.column_config.NumberColumn("🏦 Raw Closedness Score", format="%.2f")
    })

    st.divider()
    st.markdown("#### Variable Definitions")
    st.markdown("""
    * **SoV Index Score:** A composite 0-100 score determined by our PCA algorithm. It measures a population's macroeconomic necessity for unbrokered assets (100 = Highest Necessity).
    * **Regulation (0-8):** Tracks the maturity, legality, and comprehensiveness of a nation's formal digital asset frameworks (Atlantic Council).
    * **Adoption Rank:** Real-world utility and adoption of digital assets by everyday retail users (Chainalysis). *Rank 1 = Highest Adoption.*
    * **Inflation (%):** The annual percentage change in the cost of domestically manufactured goods and services via the GDP deflator (World Bank).
    * **Raw Closedness Score:** Measures a country's capital controls and regulatory restrictions on cross-border financial transactions. Adapted from the Chinn-Ito Financial Openness Index, where higher positive scores denote strict capital controls and closed markets.
    """)

# ==========================================
# TAB 3: MACRO ARCHETYPES (INTERACTIVE)
# ==========================================
with tab3:
    st.header("Propensity Archetypes in Quadrants")
    st.markdown("By mapping the **Store of Value Necessity Index** (Y-Axis) against the **Formal Regulatory Framework Score** (X-Axis), we transition from abstract macroeconomic theory to an actionable geopolitical strategy map.")

    fig_quad = px.scatter(
        df, x='regulation_jittered', y='Index_Score', color='Archetype',
        color_discrete_map=color_map, hover_name='Country_Flag',
        text='ISO Code',
        labels={'Index_Score': 'Index Score', 'regulation': 'Regulation', 'Inflation': 'Inflation (%)', 'Financial_Closedness': 'Financial Closedness', 'Crypto_Adoption_Rank': 'Crypto Adoption Rank', 'Archetype': 'Archetype'},
        hover_data={'regulation_jittered': False, 'Archetype': True, 'Index_Score': ':.1f', 'regulation': ':.1f', 'Inflation': ':.1f', 'Financial_Closedness': ':.2f', 'Crypto_Adoption_Rank': True},
        size_max=15
    )

    fig_quad.update_traces(
        textposition='top center',
        textfont=dict(size=10, color='#2C3E50'),
        marker=dict(size=14, line=dict(width=1, color='black')), 
        opacity=0.85
    )

    fig_quad.add_annotation(x=2.0, y=102, text="<b>Sovereign Controllers</b>", showarrow=False, font=dict(color="#e74c3c", size=15))
    fig_quad.add_annotation(x=6.0, y=102, text="<b>Leapfroggers</b>", showarrow=False, font=dict(color="#2ecc71", size=15))
    fig_quad.add_annotation(x=2.0, y=-2, text="<b>Low Demand Economies</b>", showarrow=False, font=dict(color="#9b59b6", size=15))
    fig_quad.add_annotation(x=6.0, y=-2, text="<b>Financial Hubs</b>", showarrow=False, font=dict(color="#3498db", size=15))

    fig_quad.update_layout(
        xaxis_title="<b>Formal Regulatory Framework Score (0-8)</b>",
        yaxis_title="<b>Store of Value Necessity Index (0-100)</b>",
        xaxis=dict(range=[-0.5, 8.5], tickmode='linear', tick0=0, dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
        plot_bgcolor="white", height=700, legend_title_text="Macro Archetypes", margin=dict(t=30, b=30, l=30, r=30)
    )

    st.plotly_chart(fig_quad, use_container_width=True)

    st.divider()
    
    st.subheader("Archetype Breakdown")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔴 Sovereign Controllers")
        st.markdown("""
        **Vibe: Survival/Control + State-Driven** These jurisdictions feature a high macroeconomic necessity for unbrokered assets (driven by inflation or capital controls), generating massive organic market demand from citizens. However, adoption is actively suppressed or tightly controlled through strict governmental frameworks. The state recognizes the utility of decentralized technology (often to bypass Western financial rails like SWIFT), but implements it strictly via top-down surveillance models (like CBDCs) rather than open, permissionless frameworks.
        """)
        
        st.markdown("### 🟣 Low Demand Economies")
        st.markdown("""
        **Vibe: Market Indifference** These regions sit at the intersection of low macroeconomic distress and low regulatory clarity. With relatively stable local currencies and accessible traditional banking, everyday citizens lack the acute "survival" catalyst needed to organically adopt unbrokered digital assets. Because the grassroots demand is low, local governments have little incentive or urgency to proactively draft comprehensive digital asset frameworks.
        """)

    with c2:
        st.markdown("### 🟢 Leapfroggers")
        st.markdown("""
        **Vibe: Survival + Market-Driven** Leapfroggers are environments experiencing severe fiat devaluation or extreme financial closedness. In these regions, unbrokered assets are not a speculative vehicle, but a literal "life raft." Everyday citizens bypass failing legacy banking systems entirely, organically adopting stablecoins and decentralized rails to protect their wealth. Crucially, the government responds to this undeniable grassroots reality by drafting accommodating regulations to capture, rather than combat, the capital flow.
        """)

        st.markdown("### 🔵 Financial Hubs")
        st.markdown("""
        **Vibe: Optimization + Institutional Arbitrage** These are wealthy, stable, financial hubs. Because inflation is low and capital mobility is high, retail demand for "life raft" crypto is negligible. Instead, the push for tokenization in these jurisdictions is entirely institutional. Governments here write crystal-clear, proactive regulations designed to lure global capital and traditional finance (TradFi) institutions seeking efficiency gains, operational optimization, and jurisdictional arbitrage.
        """)

# ==========================================
# TAB 4: POLICY SIMULATOR (WHAT-IF)
# ==========================================
with tab4:
    st.header("The \"What-If\" Simulator")
    st.markdown("""
    Our weights were determined by **Principal Component Analysis (PCA)**. PCA is a machine learning algorithm that mathematically discovers the true variance within the global economy to automatically weight each variable without human bias. 
    
    To see the effects of changes in these inputs, select a country and dynamically shift its core parameters—or override the global PCA weights—to see how these changes mathematically redefine that country within the global landscape.
    """)
    st.divider()
    
    sim_country = st.selectbox("Select Country to Simulate", sorted(df['Country'].unique()), key="sim_c")
    r_sim = df[df['Country'] == sim_country].iloc[0]
    
    # ---------------------------------------------------------
    # TOP SECTION: CONTROL PANELS SIDE-BY-SIDE
    # ---------------------------------------------------------
    col_shock, col_weights = st.columns(2)
    
    with col_shock:
        st.subheader("1. Shock the System")
        st.caption(f"Adjust the specific metrics for {sim_country}.")
        
        sim_reg = st.slider("Formal Regulation Framework (0-8)", 0.0, 8.0, float(r_sim['regulation']), 0.5,
                            help="Tracks the maturity, legality, and comprehensiveness of a nation's formal digital asset frameworks (Atlantic Council).")
        
        sim_inf = st.slider("Inflation (%)", -5.0, 150.0, float(r_sim['Inflation']), 1.0,
                            help="The annual percentage change in the cost of domestically manufactured goods and services via the GDP deflator (World Bank).")
        
        sim_close = st.slider("Capital Controls / Closedness Score (-2.5 to 2.5)", -2.5, 2.5, float(r_sim['Financial_Closedness']), 0.1, 
                              help="Adapted from the Chinn-Ito Index. -2.5 represents fully open capital markets, while 2.5 represents strict capital controls (closed economies).")
        
        sim_adopt = st.slider("Crypto Adoption Rank (1 = Highest)", 1, 150, int(r_sim['Crypto_Adoption_Rank']), 1,
                              help="Real-world utility and adoption of digital assets by everyday retail users (Chainalysis). Rank 1 = Highest Adoption.")
        
    with col_weights:
        st.subheader("2. Override Global Index Weights")
        st.caption("Shift the PCA variance weights. The system caps combinations to strictly equal 100%.")
        
        w_close = st.slider("Weight: Capital Controls (%)", 0, 100, 53,
                            help="Override the PCA variance weight for Financial Closedness/Capital Controls.")
        
        w_adopt = st.slider("Weight: Grassroots Adoption (%)", 0, 100 - w_close, min(46, 100 - w_close),
                            help="Override the PCA variance weight for Grassroots Crypto Adoption.")
        
        w_inf = 100 - w_close - w_adopt
        
        # Use a metric with a tooltip to display the calculated remainder elegantly
        st.metric(label="Weight: Inflation (%)", value=f"{w_inf}%", 
                  help="Auto-calculated remainder to ensure a perfect 100% distribution. Represents the override weight for Inflation.")

    # ---------------------------------------------------------
    # MATH & ALGORITHM RECALCULATIONS
    # ---------------------------------------------------------
    wc_n, wa_n, wi_n = w_close/100.0, w_adopt/100.0, w_inf/100.0
    models = st.session_state['pca_models']
    
    all_features = df[['Financial_Closedness', 'Inflation', 'Crypto_Adoption_Rank']]
    all_scaled = models['scaler_pca'].transform(all_features)
    
    all_raw_scores = (all_scaled[:, 0] * wc_n) + (all_scaled[:, 1] * wi_n) - (all_scaled[:, 2] * wa_n)
    new_min, new_max = all_raw_scores.min(), all_raw_scores.max()

    new_features = np.array([[sim_close, sim_inf, sim_adopt]])
    new_scaled = models['scaler_pca'].transform(new_features)[0]
    new_pca_raw = (new_scaled[0] * wc_n) + (new_scaled[1] * wi_n) - (new_scaled[2] * wa_n)
    
    if new_max == new_min: new_index = 50.0
    else: new_index = ((new_pca_raw - new_min) / (new_max - new_min)) * 100
        
    new_index = np.clip(new_index, 0, 100)
    
    new_point_df = pd.DataFrame({'regulation': [sim_reg], 'Index_Score': [new_index]})
    new_cluster_scaled = models['scaler_cluster'].transform(new_point_df)
    predicted_cluster = models['kmeans'].predict(new_cluster_scaled)[0]
    new_arch = models['cluster_mapping'][predicted_cluster]

    # ---------------------------------------------------------
    # BOTTOM SECTION: FULL WIDTH CHART & METRICS
    # ---------------------------------------------------------
    st.divider()
    st.subheader(f"Projected Impact for {sim_country}")
    
    col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
    col_res1.metric("New SoV Index Score", f"{new_index:.1f}", f"{new_index - r_sim['Index_Score']:.1f}")
    col_res2.metric("New Archetype Classification", new_arch)

    df_sim_bg = df.copy()
    df_sim_bg['Sim_Raw'] = all_raw_scores
    if new_max == new_min: df_sim_bg['Sim_Index'] = 50.0
    else: df_sim_bg['Sim_Index'] = ((df_sim_bg['Sim_Raw'] - new_min) / (new_max - new_min)) * 100
        
    bg_features = df_sim_bg[['regulation', 'Sim_Index']].rename(columns={'Sim_Index': 'Index_Score'})
    bg_cluster_scaled = models['scaler_cluster'].transform(bg_features)
    bg_clusters = models['kmeans'].predict(bg_cluster_scaled)
    df_sim_bg['Sim_Archetype'] = [models['cluster_mapping'][c] for c in bg_clusters]

    fig_sim = px.scatter(
        df_sim_bg, x='regulation', y='Sim_Index', color='Sim_Archetype',
        color_discrete_map=color_map, hover_name='Country_Flag', opacity=0.3,
        size_max=10
    )
    
    fig_sim.add_trace(go.Scatter(
        x=[r_sim['regulation']], y=[r_sim['Index_Score']],
        mode='markers', marker=dict(size=14, color='white', line=dict(width=2, color='black')),
        name=f"Original {sim_country}", hoverinfo="none"
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=[sim_reg], y=[new_index],
        mode='markers', marker=dict(size=22, symbol='star', color=color_map.get(new_arch, '#000'), line=dict(width=2, color='black')),
        name=f"Simulated {sim_country}", hoverinfo="none"
    ))
    
    fig_sim.add_annotation(
        x=sim_reg, y=new_index, ax=r_sim['regulation'], ay=r_sim['Index_Score'],
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="black"
    )
    
    fig_sim.update_layout(
        xaxis_title="<b>Regulation Score</b>", yaxis_title="<b>Index Score</b>",
        xaxis=dict(range=[-0.5, 8.5]), yaxis=dict(range=[-5, 105]),
        plot_bgcolor="white", height=600, margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_sim, use_container_width=True)

# ==========================================
# TAB 5: METHODOLOGY
# ==========================================
with tab5:
    st.header("Methodology")
    st.markdown("This dashboard relies on a rigorous combination of structured data sourcing, dimensionality reduction (PCA), and unsupervised machine learning to classify the global macroeconomic landscape.")
    
    st.markdown("### 1. Data Sourcing")
    st.markdown("""
    The Tokenized Equities Propensity Index synthesizes data from five premier global institutions to ensure a comprehensive macroeconomic snapshot:
    * **Grassroots Crypto Adoption:** Sourced from *Chainalysis*. Evaluates the real-world utility and adoption of digital assets by everyday retail users (ranked 1 to 150).
    * **Financial Closedness:** Adapted from the *Chinn-Ito Financial Openness Index*. The index measures a country's capital controls and regulatory restrictions on cross-border financial transactions. The score is mathematically inverted to measure "Closedness."
    * **Inflation (GDP Deflator):** Sourced from the *World Bank*. Measures the annual percentage change in the cost of all newly produced, domestically manufactured goods and services.
    * **Regulatory Frameworks:** Sourced from the *Atlantic Council*. Tracks the maturity, legality, and comprehensiveness of a nation's formal digital asset frameworks, scored on a continuous scale of 0 to 8.
    """)

    st.divider()

    st.markdown("### 2. Principal Component Analysis (PCA)")
    st.markdown("""
    To measure a population's intrinsic, macro-driven necessity for unbrokered assets (the **Store of Value Necessity Index**), we employed a Principal Component Analysis (PCA). 
    
    Instead of manually applying arbitrary weights to our variables to fit a narrative, PCA serves as an **unsupervised machine learning algorithm** that mathematically discovers the true variance within the global economy.
    
    1. **Standardization:** We first passed the variables (`Financial_Closedness`, `Inflation`, and `Crypto_Adoption_Rank`) through a `StandardScaler` to ensure metrics with different units (percentages, ranks, indices) were measured equally.
    2. **Variance Mapping:** The algorithm drew a line of "maximum variance" through the dataset, dynamically weighting each variable based on its actual impact on the global landscape. 
    3. **Index Generation:** The raw PCA scores were then normalized onto a clean 0 to 100 scale, plotted on the Y-Axis of our visual frameworks. The final data-driven weights assigned by the algorithm were:
        * **Financial Closedness:** 52.51%
        * **Grassroots Crypto Adoption:** 46.80%
        * **Inflation:** 0.70%
    """)

    st.divider()

    st.markdown("### 3. K-Means Clustering (Archetype Classification)")
    st.markdown("""
    To prevent human bias when categorizing countries into the four geopolitical archetypes (Financial Hubs, Leapfroggers, Sovereign Controllers, Low Demand Economies), we utilized **K-Means Clustering**.
    
    * **The Algorithm:** We instructed a K-Means algorithm to evaluate the two-dimensional space mapping our PCA Index Score (Y-Axis) against the Formal Regulatory Score (X-Axis). 
    * **Standardized Distances:** Before clustering, both axes were standardized. This ensured that the 0-100 scale of the Y-axis did not mathematically overpower the 0-8 scale of the X-axis during distance calculations.
    * **Centroid Identification:** By setting the algorithm to find exactly four clusters ($k=4$), it calculated the spatial distance between every single country, organically grouping them based on mathematical density and proximity to one of four algorithmic centroids.
    """)
