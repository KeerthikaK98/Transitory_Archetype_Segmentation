# app_choropleth.py
import os, numpy as np, pandas as pd, geopandas as gpd, pydeck as pdk, streamlit as st
from sklearn.preprocessing import MinMaxScaler
from utils import (
    load_poi_folder, load_mbta_gtfs, load_cbg_shp,
    retail_cats,  
    archetype_palette,
    compute_transit_proximity, normalize_weights, score_features,
    run_kmeans_and_label, aggregate_to_cbg, pct_short
)

st.set_page_config(page_title="MA CBG Retail Archetypes — Choropleth", layout="wide")
st.title("Transitory Archetype Segmentation — Choropleth (MA CBGs)")

# ---------- sidebar ----------
st.sidebar.header("Paths")
POI_DIR  = st.sidebar.text_input("POI CSV folder", r"C:\Users\dines\OneDrive\Documents\XN Project\_git\data\poi")
CBG_DIR  = st.sidebar.text_input("CBG shapefile folder or .shp", r"C:\Users\dines\OneDrive\Documents\XN Project\_git\data\shapes\tl_2020_25_bg")
GTFS_DIR = st.sidebar.text_input("GTFS folder", r"C:\Users\dines\OneDrive\Documents\XN Project\_git\data\gtfs")

st.sidebar.header("Weights (auto-normalized)")
w_short = st.sidebar.slider("% Short Visits",      0.0, 1.0, 0.30, 0.05)
w_vpv   = st.sidebar.slider("Visit per Visitor",   0.0, 1.0, 0.30, 0.05)
w_tp    = st.sidebar.slider("Transit Proximity",   0.0, 1.0, 0.20, 0.05)
w_raw   = st.sidebar.slider("Raw Visit Counts",    0.0, 1.0, 0.20, 0.05)
weights = normalize_weights(w_short, w_vpv, w_tp, w_raw)

#st.sidebar.header("Clustering")
#k    = st.sidebar.slider("k (clusters)", 2, 8, 4, 1)
k=4
#seed = st.sidebar.number_input("Random seed", value=42, step=1)
seed = 42

# ---------- load data ----------
poi_raw = load_poi_folder(POI_DIR)
cbg     = load_cbg_shp(CBG_DIR)
stops   = load_mbta_gtfs(
    os.path.join(GTFS_DIR, "stops.txt"),
    os.path.join(GTFS_DIR, "routes.txt"),
    os.path.join(GTFS_DIR, "trips.txt"),
    os.path.join(GTFS_DIR, "stop_times.txt"),
)

if poi_raw.empty or cbg.empty:
    st.stop()

# ---------- features ----------
poi = poi_raw[poi_raw['top_category'].isin(retail_cats)].dropna(subset=['raw_visit_counts','latitude','longitude']).copy()
poi['percent_short_visits'] = poi['bucketed_dwell_times'].apply(pct_short)
poi['geometry'] = gpd.points_from_xy(poi['longitude'], poi['latitude'])
poi_gdf = gpd.GeoDataFrame(poi, geometry='geometry', crs='EPSG:4326').to_crs(3857)

with st.spinner("Computing transit proximity..."):
    poi_gdf['transit_distance_m'] = compute_transit_proximity(poi_gdf, stops)

scaler = MinMaxScaler()
poi_gdf['transit_proximity_score'] = 1 - scaler.fit_transform(poi_gdf[['transit_distance_m']])

# dynamic scoring + clustering
poi_gdf = score_features(poi_gdf, weights)
poi_gdf, _, _ = run_kmeans_and_label(poi_gdf, k=k, seed=seed)

# Ensure CBGs are assigned to POIs
cbg = cbg.to_crs(3857)
if 'index_right' in cbg.columns:
    cbg = cbg.drop(columns='index_right')
if 'index_right' in poi_gdf.columns:
    poi_gdf = poi_gdf.drop(columns='index_right')

poi_gdf = gpd.sjoin(poi_gdf, cbg[['poi_cbg', 'geometry']], how='left', predicate='within')

# ---------- aggregate to CBG + choropleth ----------
cbg_full = aggregate_to_cbg(poi_gdf, cbg.to_crs(3857))
cbg_disp = cbg_full.to_crs(4326).copy()

# build geojson for pydeck
geojson = {"type": "FeatureCollection", "features": []}
for _, r in cbg_disp.iterrows():
    if r.geometry is None or r.geometry.is_empty:
        continue
    color = archetype_palette.get(r.get('cbg_archetype'), [180,180,180])
    geojson["features"].append({
        "type": "Feature",
        "properties": {
            "poi_cbg": r['poi_cbg'],
            "archetype": r.get('cbg_archetype'),
            "transitory": round(float(r.get('transitory_score') or 0), 3),
            "color": color
        },
        "geometry": gpd.GeoSeries([r.geometry]).__geo_interface__['features'][0]['geometry']
    })

st.subheader("CBG Choropleth — Dominant Archetype")
layer = pdk.Layer(
    "GeoJsonLayer",
    geojson,
    filled=True,
    stroked=False,
    get_fill_color="properties.color",
    get_line_color=[60,60,60],
    opacity=0.65,
    pickable=True
)
view = pdk.ViewState(latitude=42.1, longitude=-71.6, zoom=7)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                         tooltip={"text": "{poi_cbg}\n{archetype}\nTransitory: {transitory}"}))
st.subheader("Legend:")
for label, rgb in archetype_palette.items():
    hex_color = '#%02x%02x%02x' % tuple(rgb)
    st.markdown(f"<span style='color:{hex_color}; font-weight:bold;'>⬤</span> {label}", unsafe_allow_html=True)
