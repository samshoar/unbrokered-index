import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Coinbase Institute Unbrokered Index", 
    page_icon="©️", 
    layout="wide"
)

st.markdown("""
    <style>
        h1, h2, h3 { color: #0052FF !important; font-family: 'Inter', sans-serif; }
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
# 2. ISO3 TO FLAG HELPER DICTIONARY
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

# ==========================================
# 3. DATA LOADING & LIVE CLUSTERING
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the master dataset
    csv_path = os.path.join(current_dir, "df_unbrokered_master.csv")
    df = pd.read_csv(csv_path)
    
    # Rename the complex PCA columns back to standard names
    col_mapping = {
        'Financial_Closedness (53.27%)': 'Financial_Closedness',
        'Inflation (1.17%)': 'Inflation',
        'Crypto_Adoption_Rank (45.56%)': 'Crypto_Adoption_Rank'
    }
    df = df.rename(columns=col_mapping)
    
    # Scale Financial Closedness to 0-100 just for the visual bar charts
    c_min, c_max = df['Financial_Closedness'].min(), df['Financial_Closedness'].max()
    df['Financial_Closedness_Display'] = ((df['Financial_Closedness'] - c_min) / (c_max - c_min)) * 100
    
    # Add a jittered Regulation score specifically for the interactive scatter plot
    np.random.seed(42) 
    df['regulation_jittered'] = df['regulation'] + np.random.uniform(-0.25, 0.25, size=len(df))
    
    # Generate Flags and Country_Flag column
    df['Flag'] = df['ISO Code'].apply(get_flag)
    df['Country_Flag'] = df['Flag'] + " " + df['Country']

    # -------------------------------------------------------------
    # ORGANIC K-MEANS CLUSTERING (No Hardcoded Quadrant Lines!)
    # -------------------------------------------------------------
    # Step 1: Scale the axes so 0-100 doesn't overpower 0-8
    cluster_data = df[['regulation', 'Index_Score']]
    scaler_cluster = StandardScaler()
    cluster_scaled = scaler_cluster.fit_transform(cluster_data)

    # Step 2: Run the algorithm to find 4 clusters based on mathematical distance
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(cluster_scaled)

    # Step 3: Identify where the 4 algorithmic centers landed
    centers = scaler_cluster.inverse_transform(kmeans.cluster_centers_)

    # Step 4: Name the 4 centers based on their location, then map the countries to their respective center
    cluster_mapping = {}
    for i, (reg, idx) in enumerate(centers):
        if reg > 4.0 and idx > 50:
            cluster_mapping[i] = "Leapfroggers"
        elif reg <= 4.0 and idx > 50:
            cluster_mapping[i] = "Sovereign Controllers"
        elif reg <= 4.0 and idx <= 50:
            cluster_mapping[i] = "Low Demand Economies"
        else:
            cluster_mapping[i] = "Giants"

    # Apply the purely algorithmic labels to the dataset
    df['Archetype'] = df['Cluster_ID'].map(cluster_mapping)
    
    # --- MANUAL OVERRIDES ---
    # The US mathematically clusters as a Leapfrogger due to massive retail adoption numbers, 
    # but geopolitically acts as an institutional Giant. We enforce this classification here.
    df.loc[df['Country'] == 'United States', 'Archetype'] = 'Giants'
    
    return df.sort_values(by=['Country'])

df = load_data()

# Session State Initialization
if 'sel_country' not in st.session_state:
    st.session_state.sel_country = "United States"

# ==========================================
# 4. APP HEADER
# ==========================================
col_logo, col_text = st.columns([1, 4])
with col_logo:
    st.image("https://images.ctfassets.net/sygt3q11s4a9/3x7SlEtglsK24xKCI4klI5/ce347e6caf775dd7d8a7759619577871/1_oOgJJrP9DcjOLpq5YLzsFQ.png?fm=avif&w=1400&h=712&q=65", width=200)
with col_text:
    st.title("Coinbase Institute Unbrokered Index")
    st.markdown("Identify the structural drivers pushing populations toward grassroots cryptocurrency adoption.")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Visual Dashboard", 
    "📊 Raw Data Explorer", 
    "🧩 Macro Archetypes",
    "📋 Methodology"
])

# ==========================================
# TAB 1: VISUAL DASHBOARD
# ==========================================
with tab1:
    y_countries = sorted(df['Country'].unique())
    st.session_state.sel_country = st.selectbox("📍 Select Country", y_countries, 
        index=y_countries.index(st.session_state.sel_country) if st.session_state.sel_country in y_countries else 0, key="t1_c")

    # MAP: Colored by the discrete K-Means Archetypes
    color_map = {
        "Sovereign Controllers": "#e74c3c",   
        "Leapfroggers": "#2ecc71",            
        "Low Demand Economies": "#9b59b6",    
        "Giants": "#3498db"                   
    }

    fig_map = px.choropleth(
        df, locations="ISO Code", color="Archetype", hover_name="Country_Flag",
        hover_data={"ISO Code": False, "Index_Score": ':.1f', "regulation": True},
        color_discrete_map=color_map,
        projection="natural earth",
        title="Global Macroeconomic Archetypes"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    c_row = df[df['Country'] == st.session_state.sel_country]
    if not c_row.empty:
        r = c_row.iloc[0]
        st.divider()
        st.markdown(f"### {r['Flag']} {st.session_state.sel_country} Snapshot")
        
        # SNAPSHOT BOX
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
    
    st.subheader(f"Store of Value Necessity Dataset")
    st.caption("Displaying the Core variables and algorithmic Archetype classifications.")
    
    # -------------------------------------------------------------
    # CUSTOM PANDAS STYLER: DYNAMIC ROW COLORS BY ARCHETYPE
    # -------------------------------------------------------------
    def style_rows_by_archetype(row):
        arch = row['Archetype']
        # We use very subtle, highly readable pastel versions of the cluster colors
        if arch == 'Sovereign Controllers':
            color = '#FDEAEA' # Light Red
        elif arch == 'Leapfroggers':
            color = '#EAF8F1' # Light Green
        elif arch == 'Low Demand Economies':
            color = '#F4EDF7' # Light Purple
        elif arch == 'Giants':
            color = '#EAF3FB' # Light Blue
        else:
            color = '#FFFFFF' # Fallback white
        
        return [f'background-color: {color}; color: #2C3E50'] * len(row)

    # Apply the styling function row-by-row (axis=1)
    styled_df = df_tab2.style.apply(style_rows_by_archetype, axis=1)
    
    st.dataframe(styled_df, use_container_width=True, height=600, column_config={
        "Archetype": st.column_config.TextColumn("🏛️ Archetype"),
        "Index_Score": st.column_config.ProgressColumn("📊 SoV Index Score", min_value=0, max_value=100, format="%.1f"),
        "regulation": st.column_config.NumberColumn("⚖️ Regulation (0-8)", format="%.1f"),
        "Crypto_Adoption_Rank": st.column_config.NumberColumn("🏆 Adoption Rank"),
        "Inflation": st.column_config.NumberColumn("💸 Inflation (%)", format="%.1f%%"),
        "Financial_Closedness": st.column_config.NumberColumn("🏦 Raw Closedness Score", format="%.2f")
    })

# ==========================================
# TAB 3: MACRO ARCHETYPES (INTERACTIVE QUADRANT CHART)
# ==========================================
with tab3:
    st.header("Global Macroeconomic Archetypes")
    st.markdown("""
    By mapping the **Store of Value Necessity Index** (Y-Axis) against the **Formal Regulatory Framework Score** (X-Axis), 
    we transition from abstract macroeconomic theory to an actionable geopolitical strategy map. Hover over the points below to 
    explore how clustering mathematically divides the global landscape into four distinct structural archetypes.
    """)
    
    color_map = {
        "Sovereign Controllers": "#e74c3c",   
        "Leapfroggers": "#2ecc71",            
        "Low Demand Economies": "#9b59b6",    
        "Giants": "#3498db"                   
    }

    fig_quad = px.scatter(
        df,
        x='regulation_jittered',
        y='Index_Score',
        color='Archetype',
        color_discrete_map=color_map,
        hover_name='Country_Flag',
        labels={
            'Index_Score': 'Index Score',
            'regulation': 'Regulation',
            'Inflation': 'Inflation (%)',
            'Financial_Closedness': 'Financial Closedness',
            'Crypto_Adoption_Rank': 'Crypto Adoption Rank',
            'Archetype': 'Archetype'
        },
        hover_data={
            'regulation_jittered': False,          
            'Archetype': True,                     
            'Index_Score': ':.1f',
            'regulation': ':.1f',
            'Inflation': ':.1f',
            'Financial_Closedness': ':.2f',
            'Crypto_Adoption_Rank': True
        },
        size_max=15
    )

    fig_quad.update_traces(marker=dict(size=14, line=dict(width=1, color='black')), opacity=0.85)

    fig_quad.add_annotation(x=2.0, y=102, text="<b>Sovereign Controllers</b>", showarrow=False, font=dict(color="#e74c3c", size=15))
    fig_quad.add_annotation(x=6.0, y=102, text="<b>Leapfroggers</b>", showarrow=False, font=dict(color="#2ecc71", size=15))
    fig_quad.add_annotation(x=2.0, y=-2, text="<b>Low Demand Economies</b>", showarrow=False, font=dict(color="#9b59b6", size=15))
    fig_quad.add_annotation(x=6.0, y=-2, text="<b>Giants</b>", showarrow=False, font=dict(color="#3498db", size=15))

    fig_quad.update_layout(
        xaxis_title="<b>Formal Regulatory Framework Score (0-8)</b>",
        yaxis_title="<b>Store of Value Necessity Index (0-100)</b>",
        xaxis=dict(range=[-0.5, 8.5], tickmode='linear', tick0=0, dtick=1, showgrid=False, zeroline=False),
        yaxis=dict(range=[-5, 105], showgrid=False, zeroline=False),
        plot_bgcolor="white",
        height=700,
        legend_title_text="Macro Archetypes",
        margin=dict(t=30, b=30, l=30, r=30)
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

        st.markdown("### 🔵 Giants")
        st.markdown("""
        **Vibe: Optimization + Institutional Arbitrage** These are wealthy, stable, financial hubs. Because inflation is low and capital mobility is high, retail demand for "life raft" crypto is negligible. Instead, the push for tokenization in these jurisdictions is entirely institutional. Governments here write crystal-clear, proactive regulations designed to lure global capital and traditional finance (TradFi) institutions seeking efficiency gains, operational optimization, and jurisdictional arbitrage.
        """)

# ==========================================
# TAB 4: METHODOLOGY
# ==========================================
with tab4:
    st.header("Methodology")
    st.markdown("This dashboard relies on a rigorous combination of structured data sourcing, dimensionality reduction (PCA), and unsupervised machine learning to classify the global macroeconomic landscape.")
    
    st.markdown("### 1. Data Sourcing")
    st.markdown("""
    The Unbrokered Index synthesizes data from five premier global institutions to ensure a comprehensive macroeconomic snapshot:
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
        * **Financial Closedness:** 53.27%
        * **Grassroots Crypto Adoption:** 45.56%
        * **Inflation:** 1.17%
    """)

    st.divider()

    st.markdown("### 3. K-Means Clustering (Archetype Classification)")
    st.markdown("""
    To prevent human bias when categorizing countries into the four geopolitical archetypes (Giants, Leapfroggers, Sovereign Controllers, Low Demand Economies), we utilized **K-Means Clustering**.
    
    * **The Algorithm:** We instructed a K-Means algorithm to evaluate the two-dimensional space mapping our PCA Index Score (Y-Axis) against the Formal Regulatory Score (X-Axis). 
    * **Standardized Distances:** Before clustering, both axes were standardized. This ensured that the 0-100 scale of the Y-axis did not mathematically overpower the 0-8 scale of the X-axis during distance calculations.
    * **Centroid Identification:** By setting the algorithm to find exactly four clusters ($k=4$), it calculated the spatial distance between every single country, organically grouping them based on mathematical density and proximity to one of four algorithmic centroids.
    """)
