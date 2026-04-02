import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import statsmodels.formula.api as smf

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
        [data-testid="stMetricValue"] { color: #0052FF !important; font-weight: 800; }
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
# 2. DATA LOADING & SESSION STATE
# ==========================================
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "df_unbrokered_master.csv")
    df = pd.read_csv(csv_path)
    
    if 'Country' in df.columns:
        df = df.rename(columns={'Country': 'Country_Name'})
    
    # Standardize Financial Closedness (0-100)
    c_min, c_max = df['Financial_Closedness'].min(), df['Financial_Closedness'].max()
    df['Financial_Closedness_Display'] = ((df['Financial_Closedness'] - c_min) / (c_max - c_min)) * 100
    
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # FILTER: strictly 2020-2023 and ONLY countries with NO NAs
    df = df[(df['Year'] >= 2020) & (df['Year'] <= 2023)]
    
    # Identify core columns for NA removal to ensure 100% complete observations
    core_cols = ['Country_Name', 'Crypto_Adoption_Rank', 'Mobile_Connectivity', 
                 'Financial_Closedness', 'GDP_per_Capita_PPP', 'Inflation', 'Value_per_Capita_K']
    df = df.dropna(subset=core_cols)
    
    return df.sort_values(by=['Country_Name', 'Year'])

df = load_data()

# Session State Initialization
if 'sel_year' not in st.session_state:
    st.session_state.sel_year = 2023
if 'sel_country' not in st.session_state:
    st.session_state.sel_country = "United States"

# ==========================================
# 3. APP HEADER
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
    "🔬 Dynamic Regressions",
    "📋 Methodology"
])

# ==========================================
# TAB 1: VISUAL DASHBOARD
# ==========================================
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.sel_year = st.selectbox("📅 Year", sorted(df['Year'].unique(), reverse=True), key="t1_y")
    with c2:
        y_countries = sorted(df[df['Year'] == st.session_state.sel_year]['Country_Name'].unique())
        st.session_state.sel_country = st.selectbox("📍 Country", y_countries, 
            index=y_countries.index(st.session_state.sel_country) if st.session_state.sel_country in y_countries else 0, key="t1_c")

    df_year = df[df['Year'] == st.session_state.sel_year]
    
    fig_map = px.choropleth(
        df_year, locations="ISO Code", color="Crypto_Adoption_Rank", hover_name="Country_Name",
        hover_data={"ISO Code": False, "Crypto_Adoption_Rank": True},
        color_continuous_scale="RdYlGn_r", projection="natural earth"
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    c_row = df_year[df_year['Country_Name'] == st.session_state.sel_country]
    if not c_row.empty:
        r = c_row.iloc[0]
        st.divider()
        st.markdown(f"### {st.session_state.sel_country} Snapshot ({int(st.session_state.sel_year)})")
        st.markdown("<div class='snapshot-box'>", unsafe_allow_html=True)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Adoption Rank", f"#{int(r['Crypto_Adoption_Rank'])}")
        m2.metric("Connectivity", f"{r['Mobile_Connectivity']:.1f}")
        m3.metric("GDP/Capita", f"${r['GDP_per_Capita_PPP']:,.0f}")
        m4.metric("Inflation", f"{r['Inflation']:.1f}%")
        m5.metric("Market Cap", f"${r['Value_per_Capita_K']:.1f}K")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### Structural Comparison (Country vs. Global Average)")
        
        # Data prep for Horizontal Bar Chart
        metrics_labels = ['Mobile Connectivity', 'Inflation', 'Financial Closedness']
        metrics_keys = ['Mobile_Connectivity', 'Inflation', 'Financial_Closedness_Display']
        
        avg_vals = df_year[metrics_keys].mean()
        
        plot_data = pd.DataFrame({
            'Metric': metrics_labels * 2,
            'Value': list(r[metrics_keys].values) + list(avg_vals.values),
            'Scope': [st.session_state.sel_country]*3 + ['Global Average']*3
        })

        fig_horiz = px.bar(
            plot_data, x='Value', y='Metric', color='Scope', barmode='group',
            orientation='h',
            color_discrete_map={st.session_state.sel_country: '#0052FF', 'Global Average': '#E0E5EC'}
        )
        fig_horiz.update_layout(
            plot_bgcolor='white', 
            xaxis_title="Index Score / Percentage",
            yaxis_title="",
            legend_title="",
            margin=dict(t=10)
        )
        st.plotly_chart(fig_horiz, use_container_width=True)

# ==========================================
# TAB 2: RAW DATA EXPLORER
# ==========================================
with tab2:
    st.session_state.sel_year = st.selectbox("📅 Table Year", sorted(df['Year'].unique(), reverse=True), key="t2_y")
    
    # Explicitly dropping unwanted columns: Financial_Closedness (raw), Trends, and any residual adoption metrics
    # and ensuring we only show the clean, complete subset
    df_tab2 = df[df['Year'] == st.session_state.sel_year].copy()
    
    # Columns to exclude as requested
    cols_to_drop = ['Financial_Closedness', 'Year', 'ISO Code', 'Trend_Value']
    # Also removing "On-chain value received" if it exists in the raw master file
    if 'On-chain value received' in df_tab2.columns:
        cols_to_drop.append('On-chain value received')
        
    df_tab2 = df_tab2.drop(columns=[c for c in cols_to_drop if c in df_tab2.columns]).set_index('Country_Name')
    
    st.subheader(f"Complete Macroeconomic Dataset ({int(st.session_state.sel_year)})")
    st.caption("Displaying only countries with zero missing values (NAs) across the primary index metrics.")
    
    st.dataframe(df_tab2, use_container_width=True, height=600, column_config={
        "Crypto_Adoption_Rank": st.column_config.NumberColumn("🏆 Rank"),
        "Mobile_Connectivity": st.column_config.ProgressColumn("📱 Connectivity", min_value=0, max_value=100, format="%.1f"),
        "Financial_Closedness_Display": st.column_config.ProgressColumn("🏦 Fin. Closedness", min_value=0, max_value=100, format="%.1f"),
        "GDP_per_Capita_PPP": st.column_config.NumberColumn("💵 GDP", format="$%d"),
        "Inflation": st.column_config.ProgressColumn("💸 Inflation", min_value=df['Inflation'].min(), max_value=df['Inflation'].max(), format="%.1f%%"),
        "Value_per_Capita_K": st.column_config.NumberColumn("📈 Market Cap", format="$%.1fK")
    })

# ==========================================
# TAB 3: DYNAMIC REGRESSIONS
# ==========================================
with tab3:
    st.session_state.sel_year = st.radio("🔬 Analysis Year:", sorted(df['Year'].unique(), reverse=True), 
        index=sorted(df['Year'].unique(), reverse=True).index(st.session_state.sel_year), horizontal=True, key="t3_y")
    
    features = ['Mobile_Connectivity', 'Financial_Closedness', 'GDP_per_Capita_PPP', 'Inflation', 'Value_per_Capita_K']
    df_reg = df[df['Year'] == st.session_state.sel_year].copy()

    for col in features:
        df_reg[col] = (df_reg[col] - df_reg[col].mean()) / df_reg[col].std()
    
    model = smf.ols('Crypto_Adoption_Rank ~ ' + ' + '.join(features), data=df_reg).fit(cov_type='HC3')
    st.write(f"**Regression Output ({int(st.session_state.sel_year)})** | N = {len(df_reg)} | R² = {model.rsquared:.3f}")
    
    reg_summary = pd.DataFrame({'coef': model.params, 'P>|z|': model.pvalues, 'Lower 95%': model.conf_int()[0], 'Upper 95%': model.conf_int()[1]})
    st.table(reg_summary.style.format("{:.3f}").applymap(lambda v: 'font-weight: bold; color: #0052FF' if v < 0.05 else '', subset=['P>|z|']))

    st.divider()
    st.subheader(f"Quadrant Performance Summary ({int(st.session_state.sel_year)})")
    
    def get_quad(row):
        t = "High Tech" if row['Mobile_Connectivity'] >= 50 else "Low Tech"
        f = "Closed" if row['Financial_Closedness_Display'] >= 50 else "Open"
        return f"{t} / {f}"

    df_stats = df[df['Year'] == st.session_state.sel_year].copy()
    df_stats['Quadrant'] = df_stats.apply(get_quad, axis=1)
    stats = df_stats.groupby('Quadrant').agg({'Crypto_Adoption_Rank': ['mean', 'median', 'count']}).reset_index()
    stats.columns = ['Structural Quadrant', 'Mean Rank', 'Median Rank', 'N']
    st.table(stats.style.format({'Mean Rank': '{:.1f}', 'Median Rank': '{:.1f}'}).background_gradient(cmap="RdYlGn_r", subset=['Mean Rank']))

    # Load static images from current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for p in ["full", "active"]:
        img = os.path.join(current_dir, f"{p}_{int(st.session_state.sel_year)}_composite.png")
        if os.path.exists(img): st.image(img, use_container_width=True)

# ==========================================
# TAB 4: METHODOLOGY
# ==========================================
with tab4:
    st.header("Methodology")
    st.markdown("""
    * **Adoption Rank:** Chainalysis grassroots index (1 = Highest).
    * **Connectivity:** GSMA index (0-100).
    * **Fin. Closedness:** Inverted Chinn-Ito index scaled to 0-100.
    * **GDP/Capita:** PPP-adjusted international dollars (World Bank).
    * **Market Cap:** Stock market value per capita proxy.
    * **Data Cleaning:** This dashboard utilizes 'Complete Case' logic—only countries with 100% available data across all core variables are included in the index and regressions.
    """)