import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pycountry_convert

# --- 0. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Global Economic Resilience", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS for spacing and chart sizing
st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
</style>
""", unsafe_allow_html=True)

# --- 1. DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_and_process_data():
    try:
        df = pd.read_csv('economic_data.csv')
    except FileNotFoundError:
        st.error("File 'z' not found. Please ensure it is in the directory.")
        return None

    # Clean columns
    df.columns = [col.replace('_NoOutliers', '').strip() for col in df.columns]

    # Region Mapping
    def get_continent(code):
        try:
            if code.lower() == 'xk': return "Europe"
            c_code = pycountry_convert.country_alpha2_to_continent_code(code.upper())
            return pycountry_convert.convert_continent_code_to_continent_name(c_code)
        except:
            return "Other"

    df['Region'] = df['country_id'].apply(get_continent)
    
    # ISO-3 Mapping for Choropleth
    alpha2_to_alpha3 = pycountry_convert.map_country_alpha2_to_country_alpha3()
    
    def get_iso3(code):
        code = code.upper()
        if code == 'XK': return 'XKX' # Kosovo
        return alpha2_to_alpha3.get(code, None)

    df['iso_alpha'] = df['country_id'].apply(get_iso3)
    
    # Numeric conversion & Fill NA
    numeric_cols = ['Public Debt (% of GDP)', 'GDP Growth (% Annual)', 
                    'Current Account Balance (% GDP)', 'Inflation (CPI %)', 
                    'GDP (Current USD)', 'Government Revenue (% of GDP)', 
                    'Government Expense (% of GDP)', 'Unemployment Rate (%)',
                    'GDP per Capita (Current USD)']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- RESILIENCE INDEX CALCULATION ---
    scaler = MinMaxScaler()
    # Invert Debt and Inflation (Lower is better)
    fiscal = 1 - scaler.fit_transform(df[['Public Debt (% of GDP)']])
    monetary = 1 - scaler.fit_transform(df[['Inflation (CPI %)']])
    # Keep Growth and Balance (Higher is better)
    growth = scaler.fit_transform(df[['GDP Growth (% Annual)']])
    balance = scaler.fit_transform(df[['Current Account Balance (% GDP)']])

    df['Resilience Index'] = (
        (fiscal * 0.30) + 
        (growth * 0.30) + 
        (balance * 0.20) + 
        (monetary * 0.20)
    ) * 100

    # --- CLUSTERING ---
    features = ['Resilience Index', 'GDP Growth (% Annual)', 'Public Debt (% of GDP)']
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[features])
    
    means = df.groupby('Cluster')['Resilience Index'].mean().sort_values()
    names = ['Fragile Economy', 'Moderate Resilience', 'High Resilience']
    name_map = {idx: name for idx, name in zip(means.index, names)}
    df['Resilience Profile'] = df['Cluster'].map(name_map)

    return df

df = load_and_process_data()

if df is not None:
    
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.title("Dashboard Controls")
        
        # 1. Global Year Filter
        st.subheader("1. Time Period")
        min_y, max_y = int(df['year'].min()), int(df['year'].max())
        # Default to full range
        year_range = st.slider("Filter Years", min_y, max_y, (min_y, max_y))
        
        # 2. Global Region Filter
        st.subheader("2. Geography")
        all_regions = sorted(df['Region'].unique())
        # Default to all regions selected
        selected_regions = st.multiselect("Filter Regions", all_regions, default=all_regions)
        
        # --- CRITICAL: APPLY FILTERS GLOBALLY HERE ---
        df_filtered = df[
            (df['year'] >= year_range[0]) & 
            (df['year'] <= year_range[1]) & 
            (df['Region'].isin(selected_regions))
        ]
        
        st.divider()
        
        # 3. View Mode
        st.subheader("3. Select View")
        view_mode = st.radio(
            "Go to:", 
            [
                "Global Map & Trends", 
                "Country Deep Dive", 
                "Clusters & Analysis",
                "Rankings & Fiscal Health"
            ]
        )
        
        st.caption(f"Showing {len(df_filtered)} records for {len(df_filtered['country_name'].unique())} countries.")

    # ============================================================
    # VIEW 1: GLOBAL MAP (With Interactive Line Bar/Animation)
    # ============================================================
    if view_mode == "Global Map & Trends":
        st.title("Global Resilience Overview")
        
        # Ensure data is sorted by year for the animation slider to work correctly
        map_data = df_filtered.sort_values("year")

        if map_data.empty:
            st.warning("No data available for the selected filters.")
        else:
            st.markdown("### Resilience Index Animation")
            st.caption("Press the **Play Button (▶)** or drag the **Slider** at the bottom to view changes over time.")
            
            fig_map = px.choropleth(
                map_data,
                locations="country_name",
                locationmode="country names",
                color="Resilience Index",
                animation_frame="year", # <--- THIS ADDS THE INTERACTIVE LINE BAR
                hover_name="country_name",
                hover_data={
                    "year": False,
                    "Resilience Index": ":.2f",
                    "GDP Growth (% Annual)": ":.2f%",
                    "Public Debt (% of GDP)": ":.1f%"
                },
                color_continuous_scale="RdYlGn",
                range_color=(20, 80),
                projection="natural earth",
                height=650
            )
            
            fig_map.update_layout(
                margin={"r":0,"t":0,"l":0,"b":0},
                geo=dict(showframe=False, showcoastlines=False, showcountries=True),
            )
            st.plotly_chart(fig_map, use_container_width=True)

            st.divider()
            st.subheader("Regional Trends (Aggregated)")
            
            # Group by Year and Region based on filtered data
            region_stats = df_filtered.groupby(['year', 'Region'])[['Resilience Index', 'GDP Growth (% Annual)']].mean().reset_index()
            
            c1, c2 = st.columns(2)
            with c1:
                fig_reg = px.line(region_stats, x='year', y='Resilience Index', color='Region', 
                                title="Avg. Resilience Score by Region", markers=True, symbol="Region")
                fig_reg.update_layout(hovermode="x unified")
                st.plotly_chart(fig_reg, use_container_width=True)
                
            with c2:
                fig_gdp_reg = px.line(region_stats, x='year', y='GDP Growth (% Annual)', color='Region',
                                    title="Avg. GDP Growth by Region", markers=True, symbol="Region")
                fig_gdp_reg.update_layout(hovermode="x unified")
                st.plotly_chart(fig_gdp_reg, use_container_width=True)

    # ============================================================
    # VIEW 2: COUNTRY COMPARISON
    # ============================================================
    elif view_mode == "Country Deep Dive":
        st.title("Country Comparison")
        
        # Get list of countries ONLY from the filtered dataset (respects Region filter)
        available_countries = sorted(df_filtered['country_name'].unique())
        
        if not available_countries:
            st.error("No countries found for the selected Region/Year filters.")
        else:
            selected_countries = st.multiselect(
                "Select Countries to Compare:", 
                options=available_countries,
                default=available_countries[:3] if len(available_countries) >= 3 else available_countries[0:1]
            )
            
            if selected_countries:
                # Filter data for specific countries within the global timeframe
                country_data = df_filtered[df_filtered['country_name'].isin(selected_countries)]
                
                # KPI ROW (Latest available year in the filtered range)
                latest_year_in_range = country_data['year'].max()
                st.markdown(f"#### Snapshot ({latest_year_in_range})")
                
                cols = st.columns(len(selected_countries))
                for idx, country in enumerate(selected_countries):
                    c_data = country_data[(country_data['country_name'] == country) & (country_data['year'] == latest_year_in_range)]
                    if not c_data.empty:
                        with cols[idx]:
                            gdp_growth = c_data['GDP Growth (% Annual)'].values[0]
                            st.metric(
                                label=f"{country}",
                                value=f"{c_data['Resilience Index'].values[0]:.1f} / 100",
                                delta=f"{gdp_growth:.2f}% GDP Growth",
                                delta_color="normal"
                            )
                
                st.divider()

                # Charts Grid
                c1, c2 = st.columns(2)
                with c1:
                    fig1 = px.line(country_data, x='year', y='Resilience Index', color='country_name',
                                   title="Resilience Index History", markers=True, height=350)
                    fig1.update_layout(hovermode="x unified")
                    st.plotly_chart(fig1, use_container_width=True)
                    
                    fig2 = px.line(country_data, x='year', y='Public Debt (% of GDP)', color='country_name',
                                   title="Public Debt (% of GDP)", markers=True, height=350)
                    fig2.update_layout(hovermode="x unified")
                    st.plotly_chart(fig2, use_container_width=True)

                with c2:
                    fig3 = px.line(country_data, x='year', y='GDP Growth (% Annual)', color='country_name',
                                   title="GDP Growth Rate (%)", markers=True, height=350)
                    fig3.add_hline(y=0, line_dash="dot", line_color="white")
                    fig3.update_layout(hovermode="x unified")
                    st.plotly_chart(fig3, use_container_width=True)

                    fig4 = px.line(country_data, x='year', y='Inflation (CPI %)', color='country_name',
                                   title="Inflation Rate (%)", markers=True, height=350)
                    fig4.update_layout(hovermode="x unified")
                    st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Select countries above to visualize details.")

    # ============================================================
    # VIEW 3: CLUSTERS
    # ============================================================
    elif view_mode == "Clusters & Analysis":
        st.title("Economic Clusters & Shocks")
        
        if df_filtered.empty:
            st.error("No data available for selected filters.")
        else:
            col_anim, col_dist = st.columns([1.5, 1])
            
            with col_anim:
                st.subheader("Motion Chart: Debt vs. Growth")
                st.markdown("Press **Play** to watch how economies migrate between clusters.")
                
                # Ensure sorted by year for animation
                anim_data = df_filtered.sort_values("year")
                
                fig_anim = px.scatter(
                    anim_data,
                    x="Public Debt (% of GDP)",
                    y="GDP Growth (% Annual)",
                    animation_frame="year",    # Interactive Time Bar
                    animation_group="country_name",
                    size="GDP (Current USD)",  # Bubble size
                    color="Resilience Profile", 
                    hover_name="country_name",
                    color_discrete_map={
                        "High Resilience": "#00CC96", 
                        "Moderate Resilience": "#FFA15A", 
                        "Fragile Economy": "#EF553B"
                    },
                    size_max=50,
                    range_x=[-10, 200],
                    range_y=[-15, 15],
                    title="Economic Trajectory Animation"
                )
                st.plotly_chart(fig_anim, use_container_width=True)

            with col_dist:
                st.subheader("Ridgeline Distribution")
                metric = st.selectbox("Metric", ["GDP Growth (% Annual)", "Inflation (CPI %)", "Resilience Index"])
                
                fig_ridge = go.Figure()
                # Get years available in filter
                years_present = sorted(df_filtered['year'].unique(), reverse=True)
                
                for y in years_present:
                    y_dat = df_filtered[df_filtered['year'] == y]
                    fig_ridge.add_trace(go.Violin(
                        x=y_dat[metric],
                        name=str(y),
                        side='positive',
                        orientation='h',
                        width=1.5,
                        points=False,
                        line_color='#636EFA'
                    ))
                
                fig_ridge.update_layout(xaxis_title=metric, height=600, margin={"t":40, "b":20}, showlegend=False)
                st.plotly_chart(fig_ridge, use_container_width=True)

    # ============================================================
    # VIEW 4: RANKINGS
    # ============================================================
    elif view_mode == "Rankings & Fiscal Health":
        st.title("Rankings & Fiscal Health")
        
        if df_filtered.empty:
            st.error("No data available.")
        else:
            # Slider constrained by global filter
            rank_year = st.slider("Select Specific Year for Ranking", 
                                  int(df_filtered['year'].min()), 
                                  int(df_filtered['year'].max()), 
                                  int(df_filtered['year'].max()))
            
            rank_data = df_filtered[df_filtered['year'] == rank_year]

            rc1, rc2, rc3 = st.columns(3)
            
            with rc1:
                top_res = rank_data.nlargest(10, 'Resilience Index').sort_values('Resilience Index', ascending=True)
                fig_r1 = px.bar(top_res, x='Resilience Index', y='country_name', orientation='h', 
                                title="Top 10 Most Resilient", color='Resilience Index', color_continuous_scale='Greens')
                fig_r1.update_layout(yaxis_title=None, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_r1, use_container_width=True)
            
            with rc2:
                top_wealth = rank_data.nlargest(10, 'GDP per Capita (Current USD)').sort_values('GDP per Capita (Current USD)', ascending=True)
                fig_r2 = px.bar(top_wealth, x='GDP per Capita (Current USD)', y='country_name', orientation='h', 
                                title="Top 10 Wealthiest (Per Capita)", color='GDP per Capita (Current USD)', color_continuous_scale='Blues')
                fig_r2.update_layout(yaxis_title=None, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_r2, use_container_width=True)

            with rc3:
                top_debt = rank_data.nlargest(10, 'Public Debt (% of GDP)').sort_values('Public Debt (% of GDP)', ascending=True)
                fig_r3 = px.bar(top_debt, x='Public Debt (% of GDP)', y='country_name', orientation='h', 
                                title="Top 10 Highest Debt", color='Public Debt (% of GDP)', color_continuous_scale='Reds')
                fig_r3.update_layout(yaxis_title=None, showlegend=False, coloraxis_showscale=False)
                st.plotly_chart(fig_r3, use_container_width=True)

            st.divider()

            # Phillips Curve
            st.subheader("The Phillips Curve (Unemployment vs. Inflation)")
            st.markdown("Visualizing the economic trade-off between jobs and prices.")
            
            fig_phil = px.scatter(
                rank_data,
                x="Unemployment Rate (%)",
                y="Inflation (CPI %)",
                size="GDP (Current USD)",
                color="Region",
                hover_name="country_name",
                title=f"Global Phillips Curve ({rank_year})",
                height=500,
                size_max=60
            )
            # Add average lines
            avg_inf = rank_data['Inflation (CPI %)'].mean()
            avg_unemp = rank_data['Unemployment Rate (%)'].mean()
            fig_phil.add_hline(y=avg_inf, line_dash="dash", line_color="grey", annotation_text="Avg Inflation")
            fig_phil.add_vline(x=avg_unemp, line_dash="dash", line_color="grey", annotation_text="Avg Unemployment")
            
            st.plotly_chart(fig_phil, use_container_width=True)