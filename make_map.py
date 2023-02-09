from __future__ import annotations

import json
import os
import urllib.request
import zipfile
import typing

from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import matplotlib.figure
import matplotlib.patches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import shapely.affinity
import shapely.errors
import shapely.geometry
import shapely.ops

GPKG_DIR = "./gpkg/"
FALLBACK = "Cannot be selected"
categories = [
    "Icy Snow Pattern",
    "Polar Pattern",
    "Tundra Pattern",
    "Continental Pattern",
    "Garden Pattern",
    "Elegant Pattern",
    "Meadow Pattern",
    "Modern Pattern",
    "Marine Pattern",
    "Archipelago Pattern",
    "High Plains Pattern",
    "Sandstorm Pattern",
    "River Pattern",
    "Monsoon Pattern",
    "Savanna Pattern",
    "Sun Pattern",
    "Ocean Pattern",
    "Jungle Pattern",
    FALLBACK,
]
cmap = matplotlib.colors.ListedColormap([
    "#EDEDED",  # Icy Snow Pattern
    "#004EA2",  # Polar Pattern
    "#DDF1FB",  # Tundra Pattern
    "#FAC71C",  # Continental Pattern
    "#00893F",  # Garden Pattern
    "#7967A7",  # Elegant Pattern
    "#EE86AD",  # Meadow Pattern
    "#E60020",  # Modern Pattern
    "#00B0E4",  # Marine Pattern
    "#AF5201",  # Archipelago Pattern
    "#F39838",  # High Plains Pattern
    "#D5C9A1",  # Sandstorm Pattern
    "#C18700",  # River Pattern
    "#838688",  # Monsoon Pattern
    "#5AC7A5",  # Savanna Pattern
    "#FFF462",  # Sun Pattern
    "#9FD1F1",  # Ocean Pattern
    "#543822",  # Jungle Pattern
])


def download_gadm() -> None:
    """
    Downloads all of the necessary GADM geospatial data files.
    """
    data_gadm = pd.read_csv('data_gadm.tsv', sep='\t', header=0)
    futures = []
    if not os.path.exists(GPKG_DIR):
        os.mkdir(GPKG_DIR)
    with ThreadPoolExecutor() as e:
        for _, (code, url, level, old) in data_gadm.iterrows():
            filepath = os.path.join(GPKG_DIR, os.path.basename(url))
            if not os.path.exists(filepath):
                futures.append(e.submit(urllib.request.urlretrieve, url, filepath))
        for f in futures:
            print(f'{f.result()[0]} downloaded')


def get_data_from_file(code: str, url: str, level: int, old: bool, shift: int | float = 0) \
        -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads a GADM-style GPKG file from the local directory.

    :param code: The ISO 3166-1 alpha-3 for the country
    :param url: The original URL of the GADM-style GPKG file
    :param level: The administrative subdivision level to load
    :param old: Whether the file is from the GADM database, version 2.8
    :param shift: The number of degrees of longitude to shift the geometry by
    :return: A tuple of a GeoDataFrame of the subdivision data and a GeoDataFrame of the country-level data
    """
    filepath = os.path.join(GPKG_DIR, os.path.basename(url))
    print(f'Loading {filepath}...')
    if old:
        with zipfile.ZipFile(filepath) as zf:
            data = gpd.read_file(zf.open(f'{code}_adm.gpkg'), layer=f'{code}_adm{level}')
            data_outline = gpd.read_file(zf.open(f'{code}_adm.gpkg'), layer=f'{code}_adm0')
    elif level > 0:
        data = gpd.read_file(filepath, layer=f'ADM_ADM_{level}')
        data_outline = gpd.read_file(filepath, layer='ADM_ADM_0')
    else:
        data = gpd.read_file(filepath, layer='ADM_ADM_0')
        data_outline = data.copy()

    if shift != 0:
        shift_map(data, shift)
        shift_map(data_outline, shift)
    return data, data_outline


def load_gpkg_files(shift: int | float = 0) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads all of the GADM-style GPKG files listed in data_gadm.tsv and concatenates them into a single GeoDataFrame.

    :param shift: The number of degrees of longitude to shift the geometry by
    :return: A tuple of a dataframe of the subdivision data and a dataframe of the country-level data
    """
    df: list[gpd.GeoDataFrame] = []
    df_outline: list[gpd.GeoDataFrame] = []
    data_gadm = pd.read_csv('data_gadm.tsv', sep='\t', header=0)
    with ThreadPoolExecutor() as e:
        futures = [e.submit(get_data_from_file, code, url, level, old, shift)
                   for _, (code, url, level, old) in data_gadm.iterrows()]
        for f in futures:
            data, data_outline = f.result()
            df.append(data)
            df_outline.append(data_outline)
    return typing.cast((pd.concat(df).reset_index(drop=True), pd.concat(df_outline).reset_index(drop=True)),
                       tuple[gpd.GeoDataFrame, gpd.GeoDataFrame])


def load_areas_from_gpkg(shift: int | float = 0, save_pickle: bool = True) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads all of the GPKG files and processes them.

    :param shift: The number of degrees of longitude to shift the geometry by
    :param save_pickle: Whether to save the processed geospatial data to a pickle file
    :return: A tuple of a dataframe of the subdivision data and a dataframe of the country-level data
    """
    all_df, all_df_outline = load_gpkg_files(shift)

    print("Reading area data...")
    area_colors = pd.read_csv('data_areas.tsv', sep='\t', header=0)
    assert len(all_df) == len(area_colors)
    all_df['Category'] = area_colors['Category']

    print("Merging areas...")
    merge_all(all_df, all_df_outline)

    print("Simplifying geometry...")
    all_df['geometry'] = all_df['geometry'].simplify(0.025)  # approximately within half a pixel at the equator
    all_df_outline['geometry'] = all_df_outline['geometry'].simplify(0.025)

    if save_pickle:
        if shift == 0:
            all_df.to_pickle('all_df.pkl.gz', compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
            all_df_outline.to_pickle('all_df_outline.pkl.gz', compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
        else:
            all_df.to_pickle(f'all_df_{shift}.pkl.gz', compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
            all_df_outline.to_pickle(f'all_df_outline_{shift}.pkl.gz', compression={'method': 'gzip', 'compresslevel': 1, 'mtime': 1})
    return all_df, all_df_outline


def load_areas_from_pickle(shift: int | float = 0) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads all of the GPKG files from a saved pickle.

    :param shift: The number of degrees of longitude to shift the geometry by
    :return: A tuple of a dataframe of the subdivision data and a dataframe of the country-level data
    """
    print("Loading pickles...")
    if shift == 0:
        all_df = pd.read_pickle('all_df.pkl.gz')
        all_df_outline = pd.read_pickle('all_df_outline.pkl.gz')
    else:
        all_df = pd.read_pickle(f'all_df_{shift}.pkl.gz')
        all_df_outline = pd.read_pickle(f'all_df_outline_{shift}.pkl.gz')
    return all_df, all_df_outline


def merge(df: gpd.GeoDataFrame, old: str, new: str, level: str = 'GID_1') -> None:
    """
    Merges two geometries into a single geometry in the specified dataframe. This is an in-place method.

    :param df: The dataframe to modify
    :param old: The value in the row that will be dropped after the merge
    :param new: The value in the row that will be kept after the merge
    :param level: The column with the values to match on
    """
    if level not in df.keys() or not any(df[level] == old):
        print(f'Skipped: {old} -> {new} ({level})')
    df.loc[df[level] == new, 'geometry'] = \
        shapely.union_all([df[df[level] == old]['geometry'], df[df[level] == new]['geometry']])
    df.drop(df[df[level] == old].index, inplace=True)


def merge_all(all_df, all_df_outline) -> None:
    """
    Merges and removes various regions from the specified dataframe. This is an in-place method.

    :param all_df: The dataframe of the subdivision data
    :param all_df_outline: The dataframe of the country-level data
    """
    try:
        # Don't include bodies of water
        all_df.drop(all_df[(all_df['COUNTRY'] == 'Caspian Sea')
                           | (all_df['GID_1'] == 'NLD.6_1')  # IJsselmeer
                           | (all_df['GID_1'] == 'NLD.13_1')  # Zeeuwse meren
                           | (all_df['GID_1'] == 'NIC.10_1')  # Nicaragua/Lago Nicaragua
                           ].index, inplace=True)
        all_df_outline.drop(all_df_outline[all_df_outline['COUNTRY'] == 'Caspian Sea'].index, inplace=True)
        all_df_outline.loc[all_df_outline['GID_0'] == 'NLD', 'geometry'] = \
            shapely.union_all(all_df[all_df['GID_0'] == 'NLD']['geometry'])
        all_df_outline.loc[all_df_outline['GID_0'] == 'NIC', 'geometry'] = \
            shapely.union_all(all_df[all_df['GID_0'] == 'NIC']['geometry'])
    except KeyError:
        pass

    # Merge new areas
    merge(all_df, "ATG.2_1", "ATG.4_1")  # Antigua and Barbuda/Redonda -> Antigua and Barbuda/Saint John
    merge(all_df, "KOR.15_1", "KOR.3_1")  # South Korea/Sejong -> Chungcheongnam-do
    merge(all_df, "CHL.4_1", "CHL.15_1")  # Chile/Arica y Parinacota -> Tarapacá
    merge(all_df, "CHL.10_1", "CHL.9_1")  # Chile/Los Ríos -> Los Lagos
    merge(all_df, "CHL.13_1", "CHL.6_1")  # Chile/Ñuble -> Bío-Bío
    merge(all_df, "DOM.25_1", "DOM.20_1")  # Dominican Republic/San José de Ocoa -> Peravia
    merge(all_df, "DOM.31_1", "DOM.5_1")  # Dominican Republic/Santo Domingo -> Distrito Nacional
    # merge(all_df, "PAN.6_1", "PAN.5_1")  # Panama/Emberá -> Darién
    # merge(all_df, "PAN.10_1", None)  # Panama/Ngöbe Buglé
    merge(all_df, "PAN.11_1", "PAN.12_1")  # Panama/Panamá Oeste -> Panamá
    merge(all_df, "IND.32_1", "IND.2_1")  # India/Telangana -> India/Andhra Pradesh

    # Päijänne Tavastia is split across Eastern, Southern, and Western Finland, so merge them
    merge(all_df, "FIN.4.3_1", "FIN.1.3_1", level="GID_2")  # Finland/Southern Finland/Päijänne Tavastia -> Finland/Eastern Finland/Päijänne Tavastia
    merge(all_df, "FIN.5.5_1", "FIN.1.3_1", level="GID_2")  # Finland/Western Finland/Päijänne Tavastia -> Finland/Eastern Finland/Päijänne Tavastia

    # Do not display disputed regions separately
    merge(all_df, "ZNC", "CYP", level="GID_0")  # Northern Cyprus -> Cyprus
    merge(all_df, "XAD", "CYP", level="GID_0")  # Akrotiri and Dhekelia -> Cyprus
    # merge(all_df, "Z01.14_1", "IND.14_1")  # India/Jammu and Kashmir (all of it is disputed)
    merge(all_df, "Z02.28_1", "CHN.28_1")  # China/Xinjiang Uygur
    merge(all_df, "Z03.28_1", "CHN.28_1")  # China/Xinjiang Uygur
    merge(all_df, "Z03.29_1", "CHN.29_1")  # China/Xijang
    merge(all_df, "Z04.13_1", "IND.13_1")  # India/Himachal Pradesh
    merge(all_df, "Z05.35_1", "IND.35_1")  # India/Uttarakhand
    merge(all_df, "Z06", "PAK", level="GID_0")  # Pakistan (Siachen Glacier)
    merge(all_df, "Z07.3_1", "IND.3_1")  # India/Arunachal Pradesh
    merge(all_df, "Z08.29_1", "CHN.29_1")  # China/Xijang
    merge(all_df, "Z09.13_1", "IND.13_1")  # India/Himachal Pradesh
    merge(all_df, "Z09.35_1", "IND.35_1")  # India/Uttarakhand

    merge(all_df_outline, "ZNC", "CYP", level="GID_0")  # Northern Cyprus -> Cyprus
    merge(all_df_outline, "XAD", "CYP", level="GID_0")  # Akrotiri and Dhekelia -> Cyprus
    merge(all_df_outline, "Z01", "IND", level="GID_0")  # India/Jammu and Kashmir
    merge(all_df_outline, "Z02", "CHN", level="GID_0")  # China/Xinjiang Uygur
    merge(all_df_outline, "Z03", "CHN", level="GID_0")  # China/Xinjiang Uygur, China/Xijang
    merge(all_df_outline, "Z04", "IND", level="GID_0")  # India/Himachal Pradesh
    merge(all_df_outline, "Z05", "IND", level="GID_0")  # India/Uttarakhand
    merge(all_df_outline, "Z06", "PAK", level="GID_0")  # Pakistan (Siachen Glacier)
    merge(all_df_outline, "Z07", "IND", level="GID_0")  # India/Arunachal Pradesh
    merge(all_df_outline, "Z08", "CHN", level="GID_0")  # China/Xijang
    merge(all_df_outline, "Z09", "IND", level="GID_0")  # India/Himachal Pradesh, India/Uttarakhand


# https://stackoverflow.com/questions/58750837/set-centre-of-geopandas-map
def shift_map(df: gpd.GeoDataFrame, shift: int | float) -> None:
    """
    Shifts all of the geometry in the dataframe by the specified number of degrees of longitude.
    This is an in-place method.

    :param df: The dataframe to shift
    :param shift: The number of degrees of longitude to shift the geometry by
    """
    border = shapely.geometry.LineString([(shift - 180, 90), (shift - 180, -90)])
    df["geometry"] = df["geometry"].apply(shift_map_helper, convert_dtype=False, border=border, shift=shift)


def shift_map_helper(geo: shapely.Geometry, border: shapely.geometry.LineString, shift: int | float) \
        -> shapely.Geometry:
    """
    Shifts a single piece of geometry in the dataframe by the specified number of degrees of longitude.
    The piece of geometry will be split if it crosses the specified border,
    and will be combined into a single piece if it was previously split.

    :param geo: The piece of geometry to shift
    :param border: The line of longitude marking the left and right edges of the map
    :param shift: The number of degrees of longitude to shift the geometry by
    :return: The shifted piece of geometry
    """
    split_row = shapely.ops.split(geo, border)
    moved_map = []
    for item in split_row.geoms:
        minx, miny, maxx, maxy = item.bounds
        if minx >= shift - 180:
            moved_map.append(shapely.affinity.translate(item, xoff=-shift))
        else:
            moved_map.append(shapely.affinity.translate(item, xoff=360 - shift))
    return shapely.union_all(moved_map)


def make_map(*,
             filename: str | bytes | os.PathLike = None,
             language: str = 'en',
             format: str = 'png',
             shift: int | float = 0.0,
             plot_squares: bool = True,
             plot_areas: bool = True,
             plot_points: bool = True,
             save_pickle: bool = True,
             load_pickle: bool = True) -> None:
    """
    Creates a map

    :param filename: The filename that the generated map should be saved under
    :param language: The language that should be used for the legend
    :param format: The file format the generated map should be saved in
    :param shift: The number of degrees of longitude to shift the geometry by
    :param plot_squares: Whether to plot the underlying grid of squares
    :param plot_areas: Whether to plot the countries and regions
    :param plot_points: Whether to plot the coordinates for each region
    :param save_pickle: Whether to save the processed geospatial data to a pickle file
    :param load_pickle: Whether to load the processed geospatial data from a pickle file
    """
    print("Initializing plot...")
    fig, ax = plt.subplots()
    fig: matplotlib.figure.Figure
    ax: matplotlib.figure.Axes

    # Set font (the default matplotlib font does not have CJK characters)
    plt.rcParams.update({'font.size': 36})
    if language == 'ja':
        plt.rcParams.update({'font.family': "Noto Sans JP"})
    elif language == 'ko':
        plt.rcParams.update({'font.family': "Noto Sans KR"})
    elif language == 'zh-Hant':
        plt.rcParams.update({'font.family': "Noto Sans TC"})
    elif language == 'zh-Hans':
        plt.rcParams.update({'font.family': "Noto Sans SC"})

    # Set size and bounds of map
    fig.set_size_inches(80, 40)
    ax.set_aspect('equal')
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])

    # Legend with labels translated
    categories_local = pd.read_csv('data_local.tsv', sep='\t', header=0)[language].tolist() + [FALLBACK]
    handles = [matplotlib.patches.Patch(facecolor=color, edgecolor='black') for color in cmap.colors]
    labels = categories_local[:-1]
    ax.legend(handles, labels, loc='lower center', ncols=9)

    # Display grid every 10 degrees
    plt.xticks(np.arange(-180, 180 + 30, 30))
    plt.yticks(np.arange(-90, 90 + 30, 30))
    plt.xticks(np.arange(-180, 180 + 10, 10), minor=True)
    plt.yticks(np.arange(-90, 90 + 10, 10), minor=True)
    plt.grid(True, which='major', linewidth=0.5, color=(0.5, 0.5, 0.5, 0.5))
    plt.grid(True, which='minor', linewidth=0.5, color=(0.5, 0.5, 0.5, 0.5))

    # Hide axis ticks and labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    if plot_squares:
        print("Plotting squares...")
        squares_arr: list[list[int]] = \
            typing.cast(pd.read_csv('data_squares.tsv', sep='\t', header=None).values.tolist(), list[list[int]])
        squares_geo = gpd.GeoSeries(
            [shapely.geometry.box((x := (180 + 10 * j - 30) % 360 - 180),
                                  (y := 90 - (10 * i)),
                                  x + 10,
                                  y - 10)
             for i in range(len(squares_arr)) for j in range(len(squares_arr[i]))])
        squares_cat = pd.Series([color for row in squares_arr for color in row])
        squares_df = gpd.GeoDataFrame({
            'geometry': squares_geo,
            'Category': squares_cat,
        })
        if shift != 0:
            shift_map(squares_df, shift)
        squares_df.plot(ax=ax,
                        column='Category',
                        cmap=cmap,
                        linewidths=0,
                        edgecolors=(0.0, 0.0, 0.0, 0.0))

    if plot_areas:
        if not load_pickle:
            all_df, all_df_outline = load_areas_from_gpkg(shift, save_pickle)
        else:
            all_df, all_df_outline = load_areas_from_pickle(shift)
        
        print("Plotting areas...")
        all_df.loc[all_df['Category'] == FALLBACK].plot(
            ax=ax,
            linewidths=0,
            hatch="///",
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor=(0.0, 0.0, 0.0, 0.5)
        )
        ax = all_df.loc[all_df['Category'] != FALLBACK].plot(
            ax=ax,
            column='Category',
            categories=categories[:-1],
            cmap=cmap,
            linewidths=0.3,
            edgecolors='white')
        all_df_outline.plot(
            ax=ax,
            linewidths=1,
            facecolor=(0.0, 0.0, 0.0, 0.0),
            edgecolor='black')

    if plot_points:
        print("Plotting points...")
        with open('data_points.json', 'r', encoding='utf-8') as f:
            points = [(division['latitude'], division['longitude'], division['form'])
                      for country in json.load(f) for division in country['divisions']]
        ax.scatter(x=[((row[1] + (360 - shift) + 180) % 360) - 180 for row in points],
                   y=[row[0] for row in points],
                   c=[row[2] for row in points],
                   cmap=cmap,
                   marker='.',
                   linewidths=0.6,
                   edgecolors='black',
                   vmin=0,
                   vmax=17)

    print("Saving...")
    plt.savefig(filename if filename is not None else f'map_{language}_{shift}.{format}',
                format=format, bbox_inches='tight')

    print("Done!")


if __name__ == '__main__':
    import gc
    import sys
    download_gadm()
    for SHIFT in [0, 150]:  # 0 for a map centered on the Prime Meridian, 150 for a map centered on the Pacific
        make_map(shift=SHIFT, language="en", save_pickle=True, load_pickle=False)
        gc.collect()
        for LANGUAGE in ["de", "es", "fr", "it", "ja", "ko", "zh-Hant", "zh-Hans"]:
            make_map(shift=SHIFT, language=LANGUAGE, save_pickle=False, load_pickle=True)
            gc.collect()
    sys.exit()
