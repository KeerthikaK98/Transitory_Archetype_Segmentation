# Run: python src/eda.py
import os, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt, seaborn as sns
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from utils import (
    load_poi_folder, load_mbta_gtfs, load_cbg_shp,
    retail_cats, pct_short, compute_transit_proximity,
    normalize_weights, score_features, aggregate_to_cbg,
    run_kmeans_and_label
)

# ----------- paths -----------
POI_DIR   = "C:/Users/dines/OneDrive/Documents/XN Project/_git/data/poi"
GTFS_DIR  = "C:/Users/dines/OneDrive/Documents/XN Project/_git/data/gtfs"
CBG_DIR   = "C:/Users/dines/OneDrive/Documents/XN Project/_git/data/shapes/tl_2020_25_bg"
OUT_DIR   = "C:/Users/dines/OneDrive/Documents/XN Project/_git/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- load ------------
poi_raw = load_poi_folder(POI_DIR)
cbg     = load_cbg_shp(CBG_DIR)
stops   = load_mbta_gtfs(
    os.path.join(GTFS_DIR,"stops.txt"),
    os.path.join(GTFS_DIR,"routes.txt"),
    os.path.join(GTFS_DIR,"trips.txt"),
    os.path.join(GTFS_DIR,"stop_times.txt"),
)

print(f"POIs: {len(poi_raw):,} | CBGs: {len(cbg):,} | Stops: {len(stops):,}")

# ----------- feature engineering -----------
poi = poi_raw[poi_raw['top_category'].isin(retail_cats)].dropna(subset=['raw_visit_counts','latitude','longitude']).copy()
poi['percent_short_visits'] = poi['bucketed_dwell_times'].apply(pct_short)
poi['geometry'] = gpd.points_from_xy(poi['longitude'], poi['latitude'])
poi_gdf = gpd.GeoDataFrame(poi, geometry='geometry', crs='EPSG:4326').to_crs(3857)

cbg = cbg.to_crs(3857)
if 'index_right' in cbg.columns:
    cbg = cbg.drop(columns='index_right')

poi_gdf = gpd.sjoin(poi_gdf, cbg[['poi_cbg','geometry']], how='left', predicate='within')

# transit proximity
poi_gdf['transit_distance_m'] = compute_transit_proximity(poi_gdf, stops)
scaler = MinMaxScaler()
poi_gdf['transit_proximity_score'] = 1 - scaler.fit_transform(poi_gdf[['transit_distance_m']])

# weights (baseline) and score
weights = normalize_weights(0.30, 0.30, 0.20, 0.20)
poi_gdf = score_features(poi_gdf, weights)

poi_gdf, centers, sil_score = run_kmeans_and_label(poi_gdf, k=4, seed=42)

if 'index_right' in poi_gdf.columns:
    poi_gdf = poi_gdf.drop(columns='index_right')
    
# ----------- aggregate to CBG -----------
cbg_full = aggregate_to_cbg(poi_gdf, cbg.to_crs(3857)).to_crs(3857)

# ----------- quick EDA -----------
# correlation heatmap (POI level)
corr_cols = ['visit_per_visitor','percent_short_visits','transit_proximity_score','transitory_score','raw_visit_counts']
corr = poi_gdf[corr_cols].corr()
plt.figure(figsize=(7,5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation (POI level)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"corr_poi.png"), dpi=150)
plt.close()

# choropleths
fig, ax = plt.subplots(1,1, figsize=(9,9))
cbg_full.plot(column='transitory_score', cmap='OrRd', linewidth=0.2, edgecolor='lightgrey', legend=True, ax=ax)
ax.set_title("Transitory Score by CBG")
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"choropleth_transitory.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(1,1, figsize=(9,9))
cbg_full.plot(column='norm_transit_proximity_score', cmap='Blues', linewidth=0.2, edgecolor='lightgrey', legend=True, ax=ax)
ax.set_title("Transit Proximity (normalized) by CBG")
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"choropleth_transit.png"), dpi=150)
plt.close()

# overlay POIs colored by score
fig, ax = plt.subplots(1,1, figsize=(9,9))
cbg_full.plot(color='white', edgecolor='lightgrey', linewidth=0.2, ax=ax)
poi_gdf.plot(column='transitory_score', cmap='OrRd', markersize=4, alpha=0.7, legend=True, ax=ax)
ax.set_title("POIs colored by Transitory Score")
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"poi_overlay.png"), dpi=150)
plt.close()

# ----------- exports -----------
cbg_export = cbg_full[['poi_cbg','transitory_score',
                       'norm_percent_short_visits','norm_visit_per_visitor',
                       'norm_transit_proximity_score','norm_raw_visit_counts']].copy()
cbg_export.to_csv(os.path.join(OUT_DIR,"cbg_transitory_segments.csv"), index=False)
cbg_full.to_crs(4326).to_file(os.path.join(OUT_DIR,"cbg_transitory_segments.geojson"), driver="GeoJSON")

print("EDA complete. Figures and exports are in ./outputs")
