#%% 
# Detected RI cases
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.io import srtm, PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source as _SRTM3Source
import matplotlib.gridspec as gridspec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import re   # ← used to check whether the typhoon name already contains a year

# ------------------------------------------------------------
# 0. DEM processing
# ------------------------------------------------------------
class SRTM3Geo(_SRTM3Source):
    """Make cartopy-srtm fetch DEM at equal resolution under PlateCarree()."""
    def fetch_raster(self, projection, extent, target_resolution):
        return super().fetch_raster(ccrs.PlateCarree(), extent, target_resolution)

def shade(loc):
    """Add hillshade to DEM."""
    shaded = srtm.add_shading(loc.image, azimuth=135, altitude=15)
    return LocatedImage(shaded, loc.extent)

dem_source = PostprocessedRasterSource(SRTM3Geo(), shade)

# ------------------------------------------------------------
# 1. Pre-compute central longitude & view extent
# ------------------------------------------------------------
pad = 2.0
info = {}
for sid in detected_ri_sids:
    tv, pv = trues_by_sid[sid], preds_by_sid[sid]
    lons = np.concatenate([tv[:, 1], pv[:, 1]])
    lats = np.concatenate([tv[:, 0], pv[:, 0]])
    lon_min, lon_max = lons.min() - pad, lons.max() + pad
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    info[sid] = dict(
        center=0.5 * (lon_min + lon_max),
        extent=[lon_min, lon_max, lat_min, lat_max],
        true=(tv[:, 0], tv[:, 1], tv[:, 2]),
        pred=(pv[:, 0], pv[:, 1], pv[:, 2]),
    )

# ------------------------------------------------------------
# 2. Figure & grid
# ------------------------------------------------------------
nrows, ncols = 6, 3
fig = plt.figure(figsize=(ncols * 6, nrows * 5))      # ← width 6" × height 5"
gs = gridspec.GridSpec(
    nrows, ncols,
    width_ratios=[1] * ncols,
    height_ratios=[1] * nrows,
    wspace=0.15, hspace=0.40                           # ← slightly larger vertical spacing
)

def wrap_lon(lon, ctr):
    """Wrap longitude into (ctr-180, ctr+180] given a central longitude."""
    return ((lon - ctr + 180) % 360) + ctr - 180

# ------------------------------------------------------------
# 3. Draw each subplot
# ------------------------------------------------------------
letters = [f'({chr(97+i)})' for i in range(len(detected_ri_sids))]  # (a) … (r)

for idx, sid in enumerate(detected_ri_sids):
    r, c = divmod(idx, ncols)
    ctr  = info[sid]['center']
    ax   = fig.add_subplot(gs[r, c],
            projection=ccrs.PlateCarree(central_longitude=ctr))
    ax.set_aspect('auto')                              # fill the grid cell

    # ---------- Basemap & coastlines ----------
    ax.add_raster(dem_source, cmap='Greys', zorder=0)
    ax.coastlines('50m', lw=0.5, zorder=1)

    # ---------- View extent ----------
    ax.set_extent(info[sid]['extent'], crs=ccrs.PlateCarree())

    # ---------- Graticules ----------
    lon0, lon1, lat0, lat1 = info[sid]['extent']
    xticks = np.linspace(lon0, lon1, 5)
    yticks = np.linspace(lat0, lat1, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
    ax.tick_params(labelsize=8)

    # ---------- Data ----------
    t_lat, t_lon, t_w = info[sid]['true']
    p_lat, p_lon, p_w = info[sid]['pred']
    t_lon = wrap_lon(t_lon, ctr)
    p_lon = wrap_lon(p_lon, ctr)

    # True track
    for j in range(len(t_lat)):
        ax.scatter(t_lon[j], t_lat[j],
                   color=get_category_color(t_w[j]), edgecolors='k',
                   transform=ccrs.PlateCarree(), zorder=2)
        if j:
            ax.plot(t_lon[j-1:j+1], t_lat[j-1:j+1],
                    color='#696969', lw=.8,
                    transform=ccrs.PlateCarree(), zorder=2)
    # Predicted track
    for j in range(len(p_lat)):
        ax.scatter(p_lon[j], p_lat[j], marker='x',
                   color=get_category_color(p_w[j]),
                   transform=ccrs.PlateCarree(), zorder=2)
        if j:
            ax.plot(p_lon[j-1:j+1], p_lat[j-1:j+1],
                    color='brown', ls='--', lw=.8,
                    transform=ccrs.PlateCarree(), zorder=2)

    # ---------- Subplot title: avoid duplicate year + left aligned + larger font ----------
    year = str(sid)[:4]
    # If the name already contains "(20xx)", do not duplicate
    if re.search(r'\(\d{4}\)$', detected_ri_names[idx].strip()):
        core_title = detected_ri_names[idx].strip()
    else:
        core_title = f"{detected_ri_names[idx]} ({year})"
    title_str = f"{letters[idx]} {core_title}"

    try:                                              # Matplotlib ≥ 3.3
        ax.set_title(title_str, fontsize=12, loc='left', pad=2)
    except TypeError:                                 # backward compatibility
        ax.set_title(title_str, fontsize=12, pad=2, ha='left')

    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.grid(True, lw=.3, zorder=3)

# ------------------------------------------------------------
# 4. Global layout & legend
# ------------------------------------------------------------
fig.subplots_adjust(
    left=0.05, right=0.90, top=0.95, bottom=0.05,
    wspace=0.15, hspace=0.40
)
fig.legend(handles=legend_elements,
           loc='upper right', bbox_to_anchor=(0.98, 0.98))   # ← top-right corner
# fig.suptitle('TC Tracks (True vs Predicted) – Detected RI Cases', fontsize=14)

plt.savefig(path + 'Detect_RI_Case_withDEM.png', dpi=300)
plt.show()

#%%
# Missed RI cases
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.io import srtm, PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source as _SRTM3Source
import matplotlib.gridspec as gridspec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import re   # ← used to check whether the name already contains a year

# ------------------------------------------------------------
# 0. DEM processing
# ------------------------------------------------------------
class SRTM3Geo(_SRTM3Source):
    """Make cartopy-srtm properly fetch equal-resolution DEM under PlateCarree()."""
    def fetch_raster(self, projection, extent, target_resolution):
        # SRTM only supports PlateCarree; force fallback
        return super().fetch_raster(ccrs.PlateCarree(), extent, target_resolution)

def shade(loc):
    """Add hillshade to the DEM."""
    shaded = srtm.add_shading(loc.image, azimuth=135, altitude=15)
    return LocatedImage(shaded, loc.extent)

dem_source = PostprocessedRasterSource(SRTM3Geo(), shade)

# ------------------------------------------------------------
# 1. Pre-compute central longitude & view extent
# ------------------------------------------------------------
pad = 2.0
info = {}
for sid in missed_ri_sids:                               # ← process missed_ri_sids
    tv, pv = trues_by_sid[sid], preds_by_sid[sid]
    lons = np.concatenate([tv[:, 1], pv[:, 1]])
    lats = np.concatenate([tv[:, 0], pv[:, 0]])
    lon_min, lon_max = lons.min() - pad, lons.max() + pad
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    info[sid] = dict(
        center=0.5 * (lon_min + lon_max),
        extent=[lon_min, lon_max, lat_min, lat_max],
        true=(tv[:, 0], tv[:, 1], tv[:, 2]),
        pred=(pv[:, 0], pv[:, 1], pv[:, 2]),
    )

# ------------------------------------------------------------
# 2. Figure & grid
# ------------------------------------------------------------
nrows, ncols = 4, 3                                       # 12 cases: 4 × 3
fig = plt.figure(figsize=(ncols * 6, nrows * 5))          # width 6" × height 5"
gs = gridspec.GridSpec(
    nrows, ncols,
    width_ratios=[1] * ncols,
    height_ratios=[1] * nrows,
    wspace=0.15, hspace=0.40
)

def wrap_lon(lon, ctr):
    """Wrap longitudes into (ctr-180, ctr+180] given a central longitude."""
    return ((lon - ctr + 180) % 360) + ctr - 180

# ------------------------------------------------------------
# 3. Plot loop
# ------------------------------------------------------------
letters = [f'({chr(97+i)})' for i in range(len(missed_ri_sids))]  # (a) … (l)

for idx, sid in enumerate(missed_ri_sids):
    r, c = divmod(idx, ncols)
    ctr  = info[sid]['center']
    ax   = fig.add_subplot(gs[r, c],
            projection=ccrs.PlateCarree(central_longitude=ctr))
    ax.set_aspect('auto')

    # (a) Basemap & coastlines
    ax.add_raster(dem_source, cmap='Greys', zorder=0)
    ax.coastlines('50m', lw=0.5, zorder=1)

    # (b) View extent
    ax.set_extent(info[sid]['extent'], crs=ccrs.PlateCarree())

    # (c) Graticules
    lon0, lon1, lat0, lat1 = info[sid]['extent']
    xticks = np.linspace(lon0, lon1, 5)
    yticks = np.linspace(lat0, lat1, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
    ax.tick_params(labelsize=8)

    # (d) Data
    t_lat, t_lon, t_w = info[sid]['true']
    p_lat, p_lon, p_w = info[sid]['pred']
    t_lon = wrap_lon(t_lon, ctr)
    p_lon = wrap_lon(p_lon, ctr)

    # (e) True track
    for j in range(len(t_lat)):
        ax.scatter(t_lon[j], t_lat[j],
                   color=get_category_color(t_w[j]), edgecolors='k',
                   transform=ccrs.PlateCarree(), zorder=2)
        if j:
            ax.plot(t_lon[j-1:j+1], t_lat[j-1:j+1],
                    color='#696969', lw=.8,
                    transform=ccrs.PlateCarree(), zorder=2)

    # (f) Predicted track
    for j in range(len(p_lat)):
        ax.scatter(p_lon[j], p_lat[j], marker='x',
                   color=get_category_color(p_w[j]),
                   transform=ccrs.PlateCarree(), zorder=2)
        if j:
            ax.plot(p_lon[j-1:j+1], p_lat[j-1:j+1],
                    color='brown', ls='--', lw=.8,
                    transform=ccrs.PlateCarree(), zorder=2)

    # ---------- Subplot title: index + avoid duplicate year + left aligned + larger font ----------
    year = str(sid)[:4]
    # If the name already contains (YYYY), do not duplicate
    if re.search(r'\(\d{4}\)$', missed_ri_names[idx].strip()):
        core_title = missed_ri_names[idx].strip()
    else:
        core_title = f"{missed_ri_names[idx]} ({year})"
    title_str = f"{letters[idx]} {core_title}"

    try:                           # Matplotlib ≥ 3.3
        ax.set_title(title_str, fontsize=12, loc='left', pad=2)
    except TypeError:              # backward compatibility
        ax.set_title(title_str, fontsize=12, pad=2, ha='left')

    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.grid(True, lw=.3, linestyle='-', zorder=3)

# ------------------------------------------------------------
# 4. Layout & legend
# ------------------------------------------------------------
fig.subplots_adjust(
    left=0.05, right=0.90, top=0.95, bottom=0.05,
    wspace=0.15, hspace=0.40
)
fig.legend(handles=legend_elements,
           loc='upper right', fontsize=9, bbox_to_anchor=(0.98, 0.98))
# fig.suptitle('TC Tracks (True vs Predicted) – Missed RI Cases', fontsize=14)

plt.savefig(path + 'Missed_RI_Case_withDEM.png', dpi=300)
plt.show()


#%%
# Assume you already have:
# sid_test         (list of all test set TC SIDs)
# ri_trues_sids    (list of true RI TC SIDs)
# ri_preds_sids    (list of predicted RI TC SIDs by the model)

all_test_sids_set = set(sid_test)
ri_trues_sids_set = set(ri_trues_sids)

# If you only want “true non-RI”:
non_ri_sids_set = all_test_sids_set - ri_trues_sids_set

# If you want “true non-RI & also predicted non-RI” (i.e., TN):
# non_ri_sids_set = (all_test_sids_set - ri_trues_sids_set) & (all_test_sids_set - set(ri_preds_sids))
# or: non_ri_sids_set = all_test_sids_set - (ri_trues_sids_set | set(ri_preds_sids))

# Sort and convert to list
non_ri_sids = sorted(list(non_ri_sids_set))

print("Non-RI SIDs:")
print(non_ri_sids)
print(f"Count of non-RI SIDs: {len(non_ri_sids)}")

#%%
def get_typhoon_names_from_ids(target_ids, ty_ids, ty_names):
    """
    For each SID in target_ids, find its index in ty_ids,
    then get the corresponding name from ty_names and append year information.
    Returns a list where each element is in the format 'HAISHEN (2020)'.
    """
    result_list = []
    for sid in target_ids:
        if sid in ty_ids:
            index = ty_ids.index(sid)
            ty_name = ty_names[index]
            year = sid[0:4]  # The first 4 characters of SID usually represent the year
            formatted_name = f"{ty_name} ({year})"
            result_list.append(formatted_name)
        else:
            print(f"Typhoon name not found for ID {sid}.")
    return result_list

# You already have typhoon_ids, typhoon_names (parsed from the NetCDF file)
# Then:
non_ri_names = get_typhoon_names_from_ids(non_ri_sids, typhoon_ids, typhoon_names)
print(non_ri_names)

#%%
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from cartopy.io import srtm, PostprocessedRasterSource, LocatedImage
from cartopy.io.srtm import SRTM3Source as _SRTM3Source
import matplotlib.gridspec as gridspec
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ---------- 0. DEM processing ----------
class SRTM3Geo(_SRTM3Source):
    def fetch_raster(self, projection, extent, target_resolution):
        # SRTM only supports PlateCarree; force fallback here
        return super().fetch_raster(ccrs.PlateCarree(), extent, target_resolution)

def shade(loc):
    shaded = srtm.add_shading(loc.image, azimuth=135, altitude=15)
    return LocatedImage(shaded, loc.extent)

dem_source = PostprocessedRasterSource(SRTM3Geo(), shade)

# ---------- 1. Pre-compute central longitude & view extent ----------
pad = 2.0
info = {}
for sid in non_ri_sids:                                 # ← 30 non-RI cases
    tv, pv = trues_by_sid[sid], preds_by_sid[sid]
    lons = np.concatenate([tv[:, 1], pv[:, 1]])
    lats = np.concatenate([tv[:, 0], pv[:, 0]])
    lon_min, lon_max = lons.min() - pad, lons.max() + pad
    lat_min, lat_max = lats.min() - pad, lats.max() + pad
    info[sid] = dict(
        center=0.5 * (lon_min + lon_max),
        extent=[lon_min, lon_max, lat_min, lat_max],
        true=(tv[:, 0], tv[:, 1], tv[:, 2]),
        pred=(pv[:, 0], pv[:, 1], pv[:, 2]),
    )

# ---------- 2. Figure & grid ----------
nrows, ncols = 6, 5                                    # 30 cases: 6 × 5
fig = plt.figure(figsize=(ncols * 6, nrows * 5))       # width 6" × height 5"
gs = gridspec.GridSpec(
    nrows, ncols,
    width_ratios=[1] * ncols,
    height_ratios=[1] * nrows,
    wspace=0.15, hspace=0.40
)

def wrap_lon(lon, ctr):
    return ((lon - ctr + 180) % 360) + ctr - 180

# ---------- 3. Plot loop ----------
for idx, sid in enumerate(non_ri_sids):
    r, c = divmod(idx, ncols)
    ctr  = info[sid]['center']
    ax   = fig.add_subplot(
        gs[r, c], projection=ccrs.PlateCarree(central_longitude=ctr)
    )
    ax.set_aspect('auto')

    # (a) Basemap & coastlines
    ax.add_raster(dem_source, cmap='Greys', zorder=0)
    ax.coastlines('50m', lw=0.5, zorder=1)

    # (b) View extent
    ax.set_extent(info[sid]['extent'], crs=ccrs.PlateCarree())

    # (c) Graticules
    lon0, lon1, lat0, lat1 = info[sid]['extent']
    xticks = np.linspace(lon0, lon1, 5)
    yticks = np.linspace(lat0, lat1, 5)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format='.0f'))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format='.0f'))
    ax.tick_params(labelsize=8)

    # (d) Data & wrap
    t_lat, t_lon, t_w = info[sid]['true']
    p_lat, p_lon, p_w = info[sid]['pred']
    t_lon = wrap_lon(t_lon, ctr)
    p_lon = wrap_lon(p_lon, ctr)

    # (e) True track
    for j in range(len(t_lat)):
        ax.scatter(t_lon[j], t_lat[j],
                   color=get_category_color(t_w[j]), edgecolors='k',
                   transform=ccrs.PlateCarree(), zorder=2)
        if j:
            ax.plot(t_lon[j-1:j+1], t_lat[j-1:j+1],
                    color='#696969', lw=.8,
                    transform=ccrs.PlateCarree(), zorder=2)

    # (f) Predicted track
    for j in range(len(p_lat)):
        ax.scatter(p_lon[j], p_lat[j], marker='x',
                   color=get_category_color(p_w[j]),
                   transform=ccrs.PlateCarree(), zorder=2)
        if j:
            ax.plot(p_lon[j-1:j+1], p_lat[j-1:j+1],
                    color='brown', ls='--', lw=.8,
                    transform=ccrs.PlateCarree(), zorder=2)

    ax.set_title(non_ri_names[idx], fontsize=10)       # ← non_ri_names
    ax.set_xlabel('Lon')
    ax.set_ylabel('Lat')
    ax.grid(True, lw=.3, linestyle='-', zorder=3)

# ---------- 4. Layout & legend ----------
fig.subplots_adjust(
    left=0.05, right=0.90, top=0.95, bottom=0.05,
    wspace=0.15, hspace=0.40
)
fig.legend(handles=legend_elements,
           loc='upper right', fontsize=12, bbox_to_anchor=(0.98, 0.98))
# fig.suptitle('TC Tracks (True vs Predicted) – Non-RI Cases', fontsize=14)

plt.savefig(path + 'Non_RI_Case_withDEM.png', dpi=300)
plt.show()
