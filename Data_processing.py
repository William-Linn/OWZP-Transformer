# %%
import matplotlib.pyplot as plt
import optuna
import random
import joblib
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import xarray as xr
import os
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter 
import pandas as pd

#%%

path = '/home/zihaolin22/overall/'
ds = xr.open_dataset(path+'IBTrACS.WP.v04r01.nc')

# Training and validation years
start_year_train = 1980
end_year_train = 2023
year_train = ds['season']
storms_indices_train = np.where(
    (year_train >= start_year_train) & (year_train <= end_year_train))[0]

# Initialize storage lists
lat_list_train0, lon_list_train0, umax_list_train0, pres_list_train0 = [], [], [], []
sid_list_train0, time_list_train0 = [], []
u24_past_list_train0, u24_fut_list_train0 = [], []
ws_200_list_train0, owz_500_list_train0, owz_850_list_train0 = [], [], []
rh_700_list_train0, rh_925_list_train0, sph_925_list_train0 = [], [], []
D_lat_list_train0, D_lon_list_train0, angle_list_train0, speed_list_train0 = [], [], [], []
wind_change_list_train0, pres_change_list_train0 = [], []

# Yearly ERA5 cache (to avoid repeated loading)
year_data_cache = {}

# Loop through each storm
for i in storms_indices_train:
    umax_values_kt = ds['cma_wind'][i, :].values
    umax_values = umax_values_kt * 0.5144   # knots → m/s
    pres_values = ds['cma_pres'][i, :].values
    lat_values = ds['cma_lat'][i, :].values
    lon_values = np.mod(ds['cma_lon'][i, :].values + 360, 360)
    iso_time = ds['iso_time'][i, :].values
    sid = ds['sid'][i].values
    sid = sid.decode("utf-8") if isinstance(sid, bytes) else str(sid)
    sid = sid.replace("b'", "").replace("'", "")

    if np.all(np.isnan(umax_values)):
        print('Nan')
        continue
    
    # TC genesis: when it becomes a tropical storm
    valid_indices = np.where((umax_values > 17) & (~np.isnan(umax_values)))[0]
    print(len(valid_indices))

    if len(valid_indices) > 0:
        start_index = valid_indices[0]
        
        # TC dissipation: when it weakens to tropical depression
        end_index = np.where((np.isnan(umax_values[start_index:])) | (umax_values[start_index:] < 17))[0]
        end_index = end_index[0] + start_index if len(end_index) > 0 else len(umax_values)

        decoded_times = [t.decode("utf-8") for t in iso_time]
        pd_times = pd.to_datetime(decoded_times)
        hour_values = pd_times[start_index:end_index].hour
        # Extract only 6-hourly points (0, 6, 12, 18 UTC)
        six_hour_indices = [index for index, hour in enumerate(hour_values) if hour in [0, 6, 12, 18]]
        six_hour_indices = [start_index + index for index in six_hour_indices]

        # Extract track and intensity info
        lat_track = lat_values[six_hour_indices]
        lon_track = lon_values[six_hour_indices]
        umax = umax_values[six_hour_indices]
        pres = pres_values[six_hour_indices]
        time_track = pd_times[six_hour_indices]

        # Compute past 24h wind speed change
        u24_past = [0] * 4 + [umax[j] - umax[j-4] for j in range(4, len(umax))]
        
        # Compute future 24h wind speed change
        u24_fut = [umax[j+4] - umax[j] if j+4 < len(umax) else 0 for j in range(len(umax))]

        # Compute Δlat, Δlon, moving angle and speed
        D_lat = np.diff(lat_track, prepend=lat_track[0])
        D_lon = np.diff(lon_track, prepend=lon_track[0])

        # Moving angle (in radians, normalized to [0, 2π))
        angle = np.arctan2(D_lon, D_lat)
        angle[angle < 0] += 2 * np.pi  

        # Translation speed (per 6-hour interval)
        speed = np.sqrt(D_lat**2 + D_lon**2) / 6 
        
        # Wind speed change
        wind_change = np.diff(umax, prepend=umax[0])

        # Pressure change
        pres_change = np.diff(pres, prepend=pres[0])

        print(pres_change)

        # Load ERA5 environmental data for the corresponding year
        year = time_track[0].year
        print(year)
        if year not in year_data_cache:
            era5_file_path = f'/home/zihaolin22/year79_00//uvtr{year}_nm_new.nc'
            print(era5_file_path)
            xadv = xr.open_dataset(era5_file_path)
            year_data_cache[year] = {
                'lat': xadv['lat'].values,
                'lon': xadv['lon'].values,
                'time': xadv['valid_time'].values,
                'owz': xadv['n_ow_zta'].values,
                'rh': xadv['r'].values,
                'ws': xadv['wshr_sm'].values,
                'sph': xadv['sph'].values
            }

        # Get ERA5 fields
        lat_era5 = year_data_cache[year]['lat']
        lon_era5 = year_data_cache[year]['lon']
        time_era5 = year_data_cache[year]['time']
        owz = year_data_cache[year]['owz']
        rh = year_data_cache[year]['rh']
        ws = year_data_cache[year]['ws']
        sph = year_data_cache[year]['sph']

        # Match ERA5 to track points
        time_diff = np.abs(time_era5[:, np.newaxis] - time_track.values[np.newaxis, :])
        time_indices = np.argmin(time_diff, axis=0)

        lat_indices = np.abs(lat_era5[:, np.newaxis] - lat_track[np.newaxis, :]).argmin(axis=0)
        lon_indices = np.abs(lon_era5[:, np.newaxis] - lon_track[np.newaxis, :]).argmin(axis=0)

        ws_200_values, owz_500_values, owz_850_values = [], [], []
        rh_700_values, rh_925_values, sph_925_values = [], [], []

        for t_idx, lat_idx, lon_idx in zip(time_indices, lat_indices, lon_indices):
            # Extract 11°×11° region
            lat_slice = slice(max(0, lat_idx-5), min(lat_idx+5, len(lat_era5)-1))
            lon_slice = slice(max(0, lon_idx-5), min(lon_idx+5, len(lon_era5)-1))
            
            ws_200_region = ws[t_idx, 0, lat_slice, lon_slice]
            owz_500_region = owz[t_idx, 1, lat_slice, lon_slice]
            owz_850_region = owz[t_idx, 3, lat_slice, lon_slice]
            rh_700_region = rh[t_idx, 2, lat_slice, lon_slice]
            rh_925_region = rh[t_idx, 4, lat_slice, lon_slice]
            sph_925_region = sph[t_idx, 4, lat_slice, lon_slice]
            
            # Take averages or maxima as appropriate
            ws_200_values.append(ws_200_region.mean().item())
            owz_500_values.append(owz_500_region.max().item())
            owz_850_values.append(owz_850_region.max().item())
            rh_700_values.append(rh_700_region.mean().item())
            rh_925_values.append(rh_925_region.mean().item())
            sph_925_values.append(sph_925_region.mean().item())

        if len(lat_track) > 13:  # Ensure enough length for training
            lat_list_train0.append(lat_track)
            lon_list_train0.append(lon_track)
            umax_list_train0.append(umax)
            pres_list_train0.append(pres)
            sid_list_train0.append([sid] * len(lat_track))
            time_list_train0.append(time_track)
            u24_past_list_train0.append(u24_past)
            u24_fut_list_train0.append(u24_fut)
            
            D_lat_list_train0.append(D_lat)
            D_lon_list_train0.append(D_lon)
            angle_list_train0.append(angle)
            speed_list_train0.append(speed)
            wind_change_list_train0.append(wind_change)
            pres_change_list_train0.append(pres_change)

            ws_200_list_train0.append(ws_200_values)
            owz_500_list_train0.append(owz_500_values)
            owz_850_list_train0.append(owz_850_values)
            rh_700_list_train0.append(rh_700_values)
            rh_925_list_train0.append(rh_925_values)
            sph_925_list_train0.append(sph_925_values)

# %%
# Add steering flow, and recompute moving angle and translation speed
# using great-circle distance on the Earth for higher accuracy.
import xarray as xr
import numpy as np
import pandas as pd

# ---------- Configuration ----------
path = '/home/zihaolin22/overall/'
ibtracs_file = path + 'IBTrACS.WP.v04r01.nc'

# Typhoon year range
start_year_train = 1980
end_year_train   = 2023

# Steering-flow parameters
levels_dp = [200, 500, 700, 850]  # 200–850 hPa
bounds      = [levels_dp[0]] + [(levels_dp[i] + levels_dp[i+1]) / 2
                                for i in range(len(levels_dp)-1)] + [levels_dp[-1]]
dps         = np.diff(bounds)       # e.g. [150., 250., 175., 75.]
weights_np  = dps                   # Pressure-thickness weights
sum_weights = weights_np.sum()
R_km        = 270                   # Horizontal disk radius (km)

# ---------- Helper functions ----------
def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (Haversine). Returns meters."""
    R = 6371e3
    φ1, φ2 = np.deg2rad(lat1), np.deg2rad(lat2)
    Δφ     = np.deg2rad(lat2 - lat1)
    Δλ     = np.deg2rad(lon2 - lon1)
    a = np.sin(Δφ/2)**2 + np.cos(φ1)*np.cos(φ2)*np.sin(Δλ/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def bearing(lat1, lon1, lat2, lon2):
    """Azimuth (bearing) in radians within [0, 2π), clockwise from true north."""
    φ1, φ2 = np.deg2rad(lat1), np.deg2rad(lat2)
    Δλ     = np.deg2rad(lon2 - lon1)
    x = np.sin(Δλ) * np.cos(φ2)
    y = np.cos(φ1)*np.sin(φ2) - np.sin(φ1)*np.cos(φ2)*np.cos(Δλ)
    θ = np.arctan2(x, y)
    return (θ + 2*np.pi) % (2*np.pi)

def steering_flow_dp(u_dp, v_dp, time, lat0, lon0, R_km=270):
    """
    Simple disk-average of deep-layer mean wind within radius R_km.
    Returns (u_s, v_s).
    """
    u0 = u_dp.sel(time=time)
    v0 = v_dp.sel(time=time)
    lats2d, lons2d = np.meshgrid(u0.lat, u0.lon, indexing='ij')
    dist = haversine(lat0, lon0, lats2d, lons2d)
    mask = dist <= (R_km * 1000)
    return float(u0.where(mask).mean()), float(v0.where(mask).mean())

# ---------- Main script ----------
ds = xr.open_dataset(ibtracs_file)
year_train = ds['season'].values
storms_indices_train = np.where((year_train >= start_year_train) &
                                (year_train <= end_year_train))[0]

# Initialize storage containers
lat_list_train0, lon_list_train0 = [], []
umax_list_train0, pres_list_train0 = [], []
sid_list_train0, time_list_train0 = [], []
u24_past_list_train0, u24_fut_list_train0 = [], []
ws_200_list_train0, owz_500_list_train0, owz_850_list_train0 = [], [], []
rh_700_list_train0, rh_925_list_train0, sph_925_list_train0 = [], [], []
D_lat_list_train0, D_lon_list_train0 = [], []
angle_list_train0, speed_list_train0 = [], []
wind_change_list_train0, pres_change_list_train0 = [], []
steer_u_list_train0, steer_v_list_train0 = [], []

# Yearly ERA5 cache
year_data_cache = {}

for i in storms_indices_train:
    # ---- IBTrACS track ----
    umax_kt = ds['cma_wind'][i, :].values
    umax    = umax_kt * 0.5144
    pres    = ds['cma_pres'][i, :].values
    lat_vals= ds['cma_lat'][i, :].values
    lon_vals= np.mod(ds['cma_lon'][i, :].values + 360, 360)
    iso_time= ds['iso_time'][i, :].values
    sid     = ds['sid'][i].item()
    sid     = sid.decode() if isinstance(sid, bytes) else str(sid)

    if np.all(np.isnan(umax)):
        continue

    # Tropical-storm stage start/end
    valid_idx = np.where((umax > 17) & (~np.isnan(umax)))[0]
    if len(valid_idx) == 0:
        continue
    start_idx = valid_idx[0]
    end_temp  = np.where((np.isnan(umax[start_idx:])) |
                         (umax[start_idx:] < 17))[0]
    end_idx   = end_temp[0] + start_idx if len(end_temp) > 0 else len(umax)

    # 6-hourly points (00/06/12/18 UTC)
    times_dec = [t.decode() for t in iso_time]
    pd_times  = pd.to_datetime(times_dec)
    hrs       = pd_times[start_idx:end_idx].hour
    six_idx   = [j for j, h in enumerate(hrs) if h in [0,6,12,18]]
    six_inds  = [start_idx + j for j in six_idx]

    lat_track = lat_vals[six_inds]
    lon_track = lon_vals[six_inds]
    umax_tr   = umax[six_inds]
    pres_tr   = pres[six_inds]
    time_tr   = pd_times[six_inds]

    # ---- Difference features ----
    u24_past = [0]*4 + [umax_tr[j] - umax_tr[j-4]
                        for j in range(4, len(umax_tr))]
    u24_fut  = [umax_tr[j+4] - umax_tr[j]
                if j+4 < len(umax_tr) else 0
                for j in range(len(umax_tr))]
    D_lat    = np.diff(lat_track, prepend=lat_track[0])
    D_lon    = np.diff(lon_track, prepend=lon_track[0])
    wind_ch  = np.diff(umax_tr, prepend=umax_tr[0])
    pres_ch  = np.diff(pres_tr, prepend=pres_tr[0])

    # ---- Update: accurate angle & speed from great-circle distance; first point set to 0 ----
    n = len(lat_track)
    angle = np.zeros(n)
    speed = np.zeros(n)
    if n > 1:
        # Adjacent great-circle distances (m) and bearings (rad)
        dists  = haversine(lat_track[:-1], lon_track[:-1],
                           lat_track[1:],  lon_track[1:])
        brg    = bearing(lat_track[:-1], lon_track[:-1],
                         lat_track[1:],  lon_track[1:])
        spd    = (dists / 1000.0) / 6.0  # km/h over a 6-hour interval
        angle[1:] = brg
        speed[1:] = spd

    # ---- Load/cache ERA5 ----
    year = time_tr[0].year
    if year not in year_data_cache:
        era5_file = f'/home/zihaolin22/year79_00/uvtr{year}_nm.nc'
        xadv = xr.open_dataset(era5_file)
        lat_e = xadv['lat'].values
        lon_e = xadv['lon'].values
        t_e   = xadv['valid_time'].values
        ws_a  = xadv['wshr_sm'].values
        owz_a = xadv['n_ow_zta'].values
        rh_a  = xadv['r'].values
        sph_a = xadv['sph'].values
        u_a   = xadv['u'].values
        v_a   = xadv['v'].values
        sel   = [0,1,2,3]  # 200, 500, 700, 850 hPa indices
        u_sel = u_a[:, sel, :, :]
        v_sel = v_a[:, sel, :, :]
        # Deep-layer mean via pressure-thickness weighting
        u_dp  = np.tensordot(u_sel, weights_np, axes=(1,0)) / sum_weights
        v_dp  = np.tensordot(v_sel, weights_np, axes=(1,0)) / sum_weights

        year_data_cache[year] = dict(lat=lat_e, lon=lon_e, time=t_e,
                                     ws=ws_a, owz=owz_a, rh=rh_a, sph=sph_a,
                                     u_dp=u_dp, v_dp=v_dp)

    cache    = year_data_cache[year]
    lat_e    = cache['lat']; lon_e = cache['lon']; t_e = cache['time']
    ws_a     = cache['ws'];  owz_a = cache['owz']
    rh_a     = cache['rh'];  sph_a = cache['sph']
    u_dp_arr = cache['u_dp']; v_dp_arr = cache['v_dp']

    # Match indices to ERA5 grids
    t_diff = np.abs(t_e[:, None] - time_tr.values[None, :])
    ti     = np.argmin(t_diff, axis=0)
    li     = np.abs(lat_e[:, None] - lat_track[None, :]).argmin(axis=0)
    lo     = np.abs(lon_e[:, None] - lon_track[None, :]).argmin(axis=0)

    ws200_vals, owz500_vals, owz850_vals = [], [], []
    rh700_vals, rh925_vals, sph925_vals = [], [], []
    steer_u_vals, steer_v_vals = [], []

    for j, (t_idx, lat_idx, lon_idx) in enumerate(zip(ti, li, lo)):
        # Extract a local 11° × 11° box centered at the nearest grid point
        ls  = slice(max(0, lat_idx-5),  min(lat_idx+5,  len(lat_e)-1))
        lo2 = slice(max(0, lon_idx-5),  min(lon_idx+5,  len(lon_e)-1))

        ws200_vals.append(ws_a[t_idx, 0, ls, lo2].mean().item())
        owz500_vals.append(owz_a[t_idx, 1, ls, lo2].max().item())
        owz850_vals.append(owz_a[t_idx, 3, ls, lo2].max().item())
        rh700_vals.append(rh_a[t_idx, 2, ls, lo2].mean().item())
        rh925_vals.append(rh_a[t_idx, 4, ls, lo2].mean().item())
        sph925_vals.append(sph_a[t_idx, 4, ls, lo2].mean().item())

        # Steering flow: disk-average within R_km
        region_u = u_dp_arr[t_idx, ls, lo2]
        region_v = v_dp_arr[t_idx, ls, lo2]
        lats2d, lons2d = np.meshgrid(lat_e[ls], lon_e[lo2], indexing='ij')
        dist = haversine(lat_track[j], lon_track[j], lats2d, lons2d)
        mask = dist <= (R_km * 1000)
        steer_u_vals.append(float(region_u[mask].mean()))
        steer_v_vals.append(float(region_v[mask].mean()))

    # Keep only valid tracks with sufficient length
    if len(lat_track) > 13:
        lat_list_train0.append(lat_track)
        lon_list_train0.append(lon_track)
        umax_list_train0.append(umax_tr)
        pres_list_train0.append(pres_tr)
        sid_list_train0.append([sid] * len(lat_track))
        time_list_train0.append(time_tr)
        u24_past_list_train0.append(u24_past)
        u24_fut_list_train0.append(u24_fut)
        D_lat_list_train0.append(D_lat)
        D_lon_list_train0.append(D_lon)
        angle_list_train0.append(angle)
        speed_list_train0.append(speed)
        wind_change_list_train0.append(wind_ch)
        pres_change_list_train0.append(pres_ch)
        ws_200_list_train0.append(ws200_vals)
        owz_500_list_train0.append(owz500_vals)
        owz_850_list_train0.append(owz850_vals)
        rh_700_list_train0.append(rh700_vals)
        rh_925_list_train0.append(rh925_vals)
        sph_925_list_train0.append(sph925_vals)
        steer_u_list_train0.append(steer_u_vals)
        steer_v_list_train0.append(steer_v_vals)

#%%
# Save to CSV including steering-flow components u and v
data = {
    "sid":       [item for sublist in sid_list_train0       for item in sublist],
    "time":      [item for sublist in time_list_train0      for item in sublist],
    "lat":       [item for sublist in lat_list_train0       for item in sublist],
    "lon":       [item for sublist in lon_list_train0       for item in sublist],
    "umax":      [item for sublist in umax_list_train0      for item in sublist],
    "u24_past":  [item for sublist in u24_past_list_train0  for item in sublist],
    "press":     [item for sublist in pres_list_train0      for item in sublist],

    "D_lat":       [item for sublist in D_lat_list_train0     for item in sublist],
    "D_lon":       [item for sublist in D_lon_list_train0     for item in sublist],
    "mov_angle":   [item for sublist in angle_list_train0     for item in sublist],
    "mov_speed":   [item for sublist in speed_list_train0     for item in sublist],
    "wind_change": [item for sublist in wind_change_list_train0 for item in sublist],
    "pres_change": [item for sublist in pres_change_list_train0 for item in sublist],

    "owz_850":  [item for sublist in owz_850_list_train0  for item in sublist],
    "owz_500":  [item for sublist in owz_500_list_train0  for item in sublist],
    "rh_925":   [item for sublist in rh_925_list_train0   for item in sublist],
    "rh_700":   [item for sublist in rh_700_list_train0   for item in sublist],
    "ws_200":   [item for sublist in ws_200_list_train0   for item in sublist],
    "sph_925":  [item for sublist in sph_925_list_train0  for item in sublist],
    "u24_fut":  [item for sublist in u24_fut_list_train0  for item in sublist],

    # New steering-flow components
    "u_st": [item for sublist in steer_u_list_train0 for item in sublist],
    "v_st": [item for sublist in steer_v_list_train0 for item in sublist],
}

# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv(path + 'TC80-23.csv', index=False)



#%%
path = '/Users/lzh/Desktop/'
import pandas as pd
import joblib

# Read the CSV file
# df = pd.read_csv(path+'dl_data01-21.csv')
df = pd.read_csv(path+'TC80-23.csv')

# Split the dataset by year
df['year'] = df['sid'].str[:4].astype(int)

# Split the dataset
features_train = df[(df['year'] >= 1980) & (df['year'] <= 2017)]
features_val = df[(df['year'] >= 2018) & (df['year'] <= 2019)]
features_test = df[(df['year'] >= 2020) & (df['year'] <= 2023)]

# # Split the dataset
# features_train = df[(df['year'] >= 1980) & (df['year'] <= 2021)]
# features_val = df[(df['year'] >= 2015) & (df['year'] <= 2017)]
# features_test = df[(df['year'] >= 2018) & (df['year'] <= 2021)]

# Remove the auxiliary column 'year'
features_train = features_train.drop(columns=['year'])
features_val = features_val.drop(columns=['year'])
features_test = features_test.drop(columns=['year'])

def create_sliding_window(df, window_size=5):
    """
    Group by 'sid' and build sliding windows within each group (no mixing across sids).
    Returns:
      X: array of shape (num_windows, window_size, num_features)
      y: array of shape (num_windows, 1+4)  # [u24_fut, lat_next, lon_next, umax_next, press_next]
      sid_values: list of 'sid' corresponding to each window
    """
    feature_cols = [
        'lat', 'lon', 'u_st', 'v_st', 'mov_angle', 'mov_speed',
        'umax', 'u24_past', 'press', 'owz_850', 'owz_500',
        'rh_925', 'rh_700', 'ws_200', 'sph_925'
    ]
    X, y, sid_values = [], [], []

    # Group by 'sid'
    for sid, group in df.groupby('sid', sort=False):
        group = group.reset_index(drop=True)
        # Skip this TC if it has fewer than (window_size + 1) records
        if len(group) < window_size + 1:
            continue

        for i in range(len(group) - window_size):
            window = group.iloc[i:i+window_size]
            # Build X
            X_window = window[feature_cols].values
            # Build y: take u24_fut from the last point of the window
            y_u24_fut = group.iloc[i + window_size - 1]['u24_fut']
            # 'next_point' is the point immediately after the window's last row
            next_point = group.iloc[i + window_size]
            y_next = next_point[['lat', 'lon', 'umax', 'press']].values.tolist()
            X.append(X_window)
            y.append(y_next + [y_u24_fut])
            sid_values.append(sid)

    return np.array(X), np.array(y), sid_values


# Step 1: Generate X_test and y_test from the original dataset
X_test_raw, y_test_raw, sid_test = create_sliding_window(features_test)

# Step 2: Create a scaler for y_test and save it
scaler_y_test = MinMaxScaler()
y_test_scaled = scaler_y_test.fit_transform(y_test_raw)  # It is exactly the same as y_test below
joblib.dump(scaler_y_test, path+'y_test_scaler.pkl')

# Step 3: Redo everything
# Remove 'sid' and 'time' columns, preparing for normalization
features_train_no_sid_time = features_train.drop(columns=['sid', 'time'])
features_val_no_sid_time = features_val.drop(columns=['sid', 'time'])
features_test_no_sid_time = features_test.drop(columns=['sid', 'time'])

# Initialize MinMaxScaler
scaler_train = MinMaxScaler()
scaler_val = MinMaxScaler()
scaler_test = MinMaxScaler()

# Normalize each dataset
scaled_features_train = scaler_train.fit_transform(features_train_no_sid_time)
scaled_features_val = scaler_val.fit_transform(features_val_no_sid_time)
scaled_features_test = scaler_test.fit_transform(features_test_no_sid_time)

# Convert the normalized data back to DataFrame and add back the 'sid' and 'time' columns
scaled_features_train_df = pd.DataFrame(scaled_features_train, columns=features_train_no_sid_time.columns)
scaled_features_train_df['sid'] = features_train['sid'].values
scaled_features_train_df['time'] = features_train['time'].values

scaled_features_val_df = pd.DataFrame(scaled_features_val, columns=features_val_no_sid_time.columns)
scaled_features_val_df['sid'] = features_val['sid'].values
scaled_features_val_df['time'] = features_val['time'].values

scaled_features_test_df = pd.DataFrame(scaled_features_test, columns=features_test_no_sid_time.columns)
scaled_features_test_df['sid'] = features_test['sid'].values
scaled_features_test_df['time'] = features_test['time'].values

# Ensure that the column order matches the original dataset
scaled_features_train_df = scaled_features_train_df[['sid', 'time'] + list(features_train_no_sid_time.columns)]
scaled_features_val_df = scaled_features_val_df[['sid', 'time'] + list(features_val_no_sid_time.columns)]
scaled_features_test_df = scaled_features_test_df[['sid', 'time'] + list(features_test_no_sid_time.columns)]

# Create sliding window datasets using the normalized datasets
X_train, y_train, sid_train = create_sliding_window(scaled_features_train_df)
X_val, y_val, sid_val = create_sliding_window(scaled_features_val_df)
X_test, y_test, sid_test = create_sliding_window(scaled_features_test_df)


#%%
# Convert 2D y to 3D y
y_train = y_train[:, np.newaxis, :]
y_val = y_val[:, np.newaxis, :]
y_test = y_test[:, np.newaxis, :]

path = '/Users/lzh/Desktop/'

with open(path+'train_dataset.pkl', 'wb') as f:
    pickle.dump((X_train, y_train), f)

with open(path+'val_dataset.pkl', 'wb') as f:
    pickle.dump((X_val, y_val), f)

with open(path+'test_dataset.pkl', 'wb') as f:
    pickle.dump((X_test, y_test), f)
    

#%%
path = '/Users/lzh/Desktop/'
# Load data from files
with open(path+'train_dataset.pkl', 'rb') as f:
    X_train, y_train = pickle.load(f)

with open(path+'val_dataset.pkl', 'rb') as f:
    X_val, y_val = pickle.load(f)

with open(path+'test_dataset.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)


# Load the scaler for the test set
scaler_y_test = joblib.load(path+'y_test_scaler.pkl')

# Convert to PyTorch datasets
train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

# Create DataLoader
batch_size = 128  # Can be changed to 64 or 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print the size of the datasets
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
