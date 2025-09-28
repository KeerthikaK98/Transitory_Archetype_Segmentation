import os, ast, numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------- constants --------------------
retail_cats = set([
    'Home Furnishings Stores','Electronics and Appliance Stores','Lawn and Garden Equipment and Supplies Stores',
    'Clothing Stores','Personal Care Services','Household Appliance Manufacturing','Automotive Parts, Accessories, and Tire Stores',
    'Furniture Stores','Grocery Stores','Specialty Food Stores','Beer, Wine, and Liquor Stores','Health and Personal Care Stores',
    'Shoe Stores','Jewelry, Luggage, and Leather Goods Stores','Sporting Goods, Hobby, and Musical Instrument Stores',
    'Book Stores and News Dealers','Department Stores','General Merchandise Stores, including Warehouse Clubs and Supercenters',
    'Office Supplies, Stationery, and Gift Stores','Used Merchandise Stores','Other Miscellaneous Store Retailers',
    'Offices of Other Health Practitioners','Home Health Care Services','Other Ambulatory Health Care Services',
    'Community Food and Housing, and Emergency and Other Relief Services','Special Food Services','Restaurants and Other Eating Places',
    'Apparel Accessories and Other Apparel Manufacturing','Household Appliances and Electrical and Electronic Goods Merchant Wholesalers',
    'Grocery and Related Product Merchant Wholesalers','Other Food Manufacturing'
])

archetype_palette = {
    'Commuter Hub': [33,158,188],
    'Convenience Zone': [251,133,0],
    'Destination Retail': [102,194,165],
    'Experience-Oriented': [141,160,203],
    'Non-Retail Areas': [204,204,204]
}

# -------------------- parsing & feature helpers --------------------
def fix_and_parse(val):
    if pd.isna(val): return {}
    if isinstance(val, dict): return val
    try: return ast.literal_eval(str(val).replace('""','"'))
    except Exception: return {}

def pct_short(d):
    if isinstance(d, dict):
        short = d.get('<5', 0) + d.get('5-20', 0)
        tot = sum(d.values())
        return short/tot if tot else 0.0
    return 0.0

def normalize_weights(w_short, w_vpv, w_tp, w_raw):
    s = w_short + w_vpv + w_tp + w_raw
    if s == 0: return (0.25, 0.25, 0.25, 0.25)
    return tuple(w/s for w in [w_short, w_vpv, w_tp, w_raw])

# -------------------- loaders --------------------
def load_poi_folder(folder_path: str) -> pd.DataFrame:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    chunks = []
    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        df = df[(df.get('region') == 'MA') & (df.get('closed_on').isnull())]
        df['bucketed_dwell_times'] = df['bucketed_dwell_times'].apply(fix_and_parse)
        df['visit_per_visitor'] = df['raw_visit_counts'] / df['raw_visitor_counts']
        chunks.append(df)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def load_mbta_gtfs(stops_fp, routes_fp, trips_fp, stop_times_fp) -> gpd.GeoDataFrame:
    stops = pd.read_csv(stops_fp)
    routes = pd.read_csv(routes_fp)
    trips  = pd.read_csv(trips_fp)
    stop_times = pd.read_csv(stop_times_fp)

    stop_route = stop_times.merge(trips[['trip_id','route_id']], on='trip_id', how='left') \
                           .merge(routes[['route_id','route_type']], on='route_id', how='left') \
                           .merge(stops, on='stop_id', how='left')
    stop_route = stop_route.drop_duplicates(subset='stop_id')
    mask = stop_route['route_type'].isin([1,3])  # 1=subway, 3=bus
    all_stops = stop_route.loc[mask, ['stop_id','stop_lat','stop_lon']].dropna()
    gdf = gpd.GeoDataFrame(
        all_stops,
        geometry=gpd.points_from_xy(all_stops['stop_lon'], all_stops['stop_lat']),
        crs='EPSG:4326'
    ).to_crs(3857)
    return gdf

def load_cbg_shp(cbg_dir: str) -> gpd.GeoDataFrame:
    cbg = gpd.read_file(cbg_dir)
    if 'GEOID' in cbg.columns:
        cbg = cbg.rename(columns={'GEOID':'poi_cbg'})
    elif 'cbg' in cbg.columns:
        cbg = cbg.rename(columns={'cbg':'poi_cbg'})
    return cbg

# -------------------- geospatial transforms --------------------
def compute_transit_proximity(pois_3857: gpd.GeoDataFrame, stops_3857: gpd.GeoDataFrame) -> pd.Series:
    if pois_3857.empty:
        return pd.Series(dtype="float64", index=pois_3857.index)
    if stops_3857.empty:
        return pd.Series(np.nan, index=pois_3857.index)

    if pois_3857.crs != stops_3857.crs:
        stops_3857 = stops_3857.to_crs(pois_3857.crs)

    # Keep original POI index as an explicit column
    left = pois_3857[['geometry']].reset_index().rename(columns={'index': 'orig'})

    joined = gpd.sjoin_nearest(
        left,                      # POIs with 'orig' column
        stops_3857[['geometry']],  # stops
        how='left',
        distance_col='transit_distance_m'
    )

    # Collapse potential ties: one distance per 'orig'
    dist_per_poi = joined.groupby('orig', as_index=True)['transit_distance_m'].min()

    # Align back to the original POI index (which 'orig' encodes)
    return dist_per_poi.reindex(pois_3857.index)

def score_features(df: pd.DataFrame, weights_tuple):
    feat_cols = ['percent_short_visits','visit_per_visitor','transit_proximity_score','raw_visit_counts']
    z = df[feat_cols].fillna(0).values
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(z)
    for i, c in enumerate(feat_cols):
        df[f'norm_{c}'] = norm[:, i]
    w_short, w_vpv, w_tp, w_raw = weights_tuple
    df['transitory_score'] = (
        w_short * df['norm_percent_short_visits'] +
        w_vpv   * df['norm_visit_per_visitor'] +
        w_tp    * df['norm_transit_proximity_score'] +
        w_raw   * df['norm_raw_visit_counts']
    )
    return df

# -------------------- clustering & aggregation --------------------
def run_kmeans_and_label(poi_df_norm: pd.DataFrame, k: int, seed: int):
    X = poi_df_norm[['norm_percent_short_visits','norm_visit_per_visitor',
                     'norm_transit_proximity_score','norm_raw_visit_counts']]
    km = KMeans(n_clusters=k, random_state=seed, n_init='auto')
    labels = km.fit_predict(X)
    centers = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    # static mapping; adjust if you later reorder by centroid traits
    label_map = {0:'Commuter Hub', 1:'Convenience Zone', 2:'Destination Retail', 3:'Experience-Oriented'}
    poi_df_norm['cluster'] = labels
    poi_df_norm['retail_archetype'] = poi_df_norm['cluster'].map(label_map).fillna('Experience-Oriented')
    sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    return poi_df_norm, centers, sil

def aggregate_to_cbg(poi_gdf: gpd.GeoDataFrame, cbg_3857: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    for df in [poi_gdf, cbg_3857]:
        if 'index_right' in df.columns:
            df.drop(columns='index_right', inplace=True)

    joined = gpd.sjoin(poi_gdf, cbg_3857[['poi_cbg','geometry']], how='left', predicate='within')
    dom_arch = (
        joined.groupby('poi_cbg')['retail_archetype']
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index(name='cbg_archetype')
    )
    means = (
        joined.groupby('poi_cbg')[['transitory_score',
                                    'norm_percent_short_visits',
                                    'norm_visit_per_visitor',
                                    'norm_transit_proximity_score',
                                    'norm_raw_visit_counts']]
        .mean()
        .reset_index()
    )
    cbg_full = cbg_3857.merge(dom_arch, on='poi_cbg', how='left') \
                       .merge(means, on='poi_cbg', how='left')
    return cbg_full
