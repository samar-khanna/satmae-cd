import argparse
import os
import ast
import json
import rasterio
import numpy as np
from glob import glob
from tqdm import tqdm

from math import floor, ceil
from PIL import Image
from matplotlib.path import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from util.qfabric_dataset import QFabricDataset


def annotate_json(gjson_file, json_file):
    with open(gjson_file, 'r') as f:
        g_file = json.load(f)

    with open(json_file, 'r') as f:
        file = json.load(f)

    features = {int(feat['properties']['label']): feat for feat in g_file['features']}
    shapes = {int(shape['label']): shape for shape in file['shapes']}
    # features = sorted(g_file['features'], key=lambda feat: int(feat['properties']['label']))
    # shapes = sorted(file['shapes'], key=lambda shape: int(shape['label']))

    # assert len(features) == len(shapes), \
    #     f'num feats in {gjson_file}: {len(features)}, {json_file}: {len(shapes)} do not match'

    # Set the properties from geojson to the json file
    new_shapes = []
    for label, shape in shapes.items():
        # assert label in features, \
        #     f'Label {label} not present in {gjson_file}. ' \
        #     f'Num features: {gjson_file}: {len(features)}, {json_file}: {len(shapes)}'
        if label not in features:
            continue

        feat = features[label]

        fmt_change_status = feat['properties']['change_status']
        if len(fmt_change_status) > 0:
            feat['properties']['change_status'] = ast.literal_eval(json.loads(fmt_change_status))
        else:
            feat['properties']['change_status'] = {}

        shape['properties'] = feat['properties']
        new_shapes.append(shape)

    file['shapes'] = new_shapes
    return file


def annotate_jsons(gjson_path, json_path):

    new_json_dir = json_path.replace('jsons', 'new_jsons')
    os.makedirs(new_json_dir, exist_ok=True)

    # Assume files are x.geojson and x.json, where x is some non-negative int
    gjson_files = glob(os.path.join(gjson_path, '*.geojson'))
    gjson_files = sorted(gjson_files, key=f_name_sort_key)

    json_files = glob(os.path.join(json_path, '*.json'))
    json_files = sorted(json_files, key=f_name_sort_key)

    with ThreadPoolExecutor(max_workers=8) as exec:
        future_to_file = {exec.submit(annotate_json, gjson_f, json_f): (gjson_f, json_f)
                          for gjson_f, json_f in zip(gjson_files, json_files)}

        for future in tqdm(as_completed(future_to_file), total=len(future_to_file)):
            try:
                new_json = future.result()
            except Exception as e:
                raise e

            gjson_f, json_f = future_to_file[future]
            f_name = json_f.split(os.path.sep)[-1]
            with open(os.path.join(new_json_dir, f_name), 'w') as f:
                json.dump(new_json, f, indent=2)


def gen_grid_points(x_min, x_max, y_min, y_max):
    # make a canvas with pixel coordinates
    x, y = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    x, y = x.reshape(-1), y.reshape(-1)
    # A list of all pixels in terms of indices
    coords = np.vstack((x, y)).T  # ((x_max-x_min-1)*(y_max-y_min-1), 2)
    return coords


def poly_mask(polygon_coords, w_max, h_max):
    x_min, x_max = np.min(polygon_coords[:, 0]), np.max(polygon_coords[:, 0])
    y_min, y_max = np.min(polygon_coords[:, 1]), np.max(polygon_coords[:, 1])

    x_min, x_max = max(0, floor(x_min)), min(w_max, ceil(x_max+1))
    y_min, y_max = max(0, floor(y_min)), min(h_max, ceil(y_max+1))
    valid_coords = gen_grid_points(x_min, x_max, y_min, y_max)  # ((x_max-x_min)*(y_max-y_min), 2)

    # Only need to find polygon mask within the rectangular patch containing the polygon
    path = Path(polygon_coords)
    coords_mask = path.contains_points(valid_coords)
    return coords_mask, (x_min, x_max, y_min, y_max)


def create_change_type_mask(raster_files, json_file, change_types):
    raster_file = raster_files[0]
    with rasterio.open(raster_file) as raster:
        h = raster.height
        w = raster.width

    with open(json_file, 'r') as f:
        labels = json.load(f)

    all_coords = gen_grid_points(0, w, 0, h)  # (h*w, 2)
    assert all_coords.shape == (h*w, 2)

    # all_polygons = [np.array(shape['points']) for shape in labels['shapes']]
    # label_mask = np.zeros(h * w, dtype=np.uint8)
    # with ThreadPoolExecutor(max_workers=32) as ex:
    #     future_to_shape = {ex.submit(poly_mask, poly_coords): i
    #                        for i, poly_coords in enumerate(all_polygons)}
    #
    #     for future in tqdm(as_completed(future_to_shape), total=len(future_to_shape)):
    #         try:
    #             intersection_mask, bounds = future.result()
    #         except Exception as e:
    #             raise e
    #
    #         shape_idx = future_to_shape[future]
    #
    #         label = labels['shapes'][shape_idx]['properties']['change_type']
    #         label_idx = change_types.index(label)
    #
    #         x_min, x_max, y_min, y_max = bounds
    #
    #         fill_values = label_mask[y_min:y_max, x_min:x_max]
    #         f_h, f_w = fill_values.shape
    #         fill_values = fill_values.reshape(-1)
    #         fill_values[intersection_mask] = label_idx
    #
    #         label_mask[y_min:y_max, x_min:x_max] = fill_values.reshape(f_h, f_w)

    # # ASSUME: NO OVERLAP
    label_mask = np.zeros((h, w), dtype=np.uint8)
    for shape in labels['shapes']:
        label = shape['properties']['change_type']
        label_idx = change_types.index(label)

        poly_coords = np.array(shape['points'])  # (P, 2)

        intersection_mask, bounds = poly_mask(poly_coords, w, h)
        x_min, x_max, y_min, y_max = bounds

        fill_values = label_mask[y_min:y_max, x_min:x_max]
        f_h, f_w = fill_values.shape
        fill_values = fill_values.reshape(-1)
        fill_values[intersection_mask] = label_idx

        label_mask[y_min:y_max, x_min:x_max] = fill_values.reshape(f_h, f_w)

    return label_mask.reshape(h, w)


def create_change_type_masks(raster_dir, json_dir, change_types):
    out_dir = json_dir.replace('new_jsons', 'change_type_masks')
    os.makedirs(out_dir, exist_ok=True)

    json_files = glob(os.path.join(json_dir, '*.json'))
    json_files = sorted(json_files, key=f_name_sort_key)

    def raster_sort_key(f_path):
        f_name = f_path.split(os.path.sep)[-1]
        f_id, d, date_str, _ = f_name.split('.')
        return (int(f_id), d)

    raster_files = glob(os.path.join(raster_dir, '*.tif'))
    raster_files = sorted(raster_files, key=raster_sort_key)

    assert 5*len(json_files) == len(raster_files), \
        f"Mismatch in num labels {len(json_files)}, and rasters {len(raster_files)}"

    with ThreadPoolExecutor(max_workers=8) as exec:
        future_to_mask = {
            exec.submit(
                create_change_type_mask,
                raster_files[5*i:5*i + 5],
                json_files[i],
                change_types,
            ): i
            for i in range(len(json_files))
        }

        for future in tqdm(as_completed(future_to_mask), total=len(future_to_mask)):
            try:
                mask = future.result()
            except Exception as e:
                raise e

            idx = future_to_mask[future]
            mask = Image.fromarray(mask)
            mask.save(os.path.join(out_dir, f"{idx}.png"))


def create_tile(array_file, tile_size=224, file_ext='tif'):
    if file_ext == 'tif':
        with rasterio.open(array_file) as rst:
            img = rst.read()  # (3, h, w)
            h, w = rst.height. rst.width
            c = rst.count
    elif file_ext == 'png':
        img = Image.open(array_file)
        img = np.array(img)  # (h, w)
        h, w = img.shape
        img = img.reshape(1, h, w)
        c = 1
    else:
        raise NotImplementedError

    h_cut = (h//tile_size) * tile_size
    w_cut = (w//tile_size) * tile_size
    img = img[:, :h_cut, :w_cut]

    h_t, w_t = h_cut // tile_size, w_cut // tile_size
    img = img.reshape(c, h_t, tile_size, w_t, tile_size)  # (c, h//t, t, w//t, t)
    img = np.einsum('chpwq->hwcpq', img)  # (h//t,w//t,c,t,t)
    tiles = img.reshape(h_t * w_t, c, tile_size, tile_size)  # (h//t*w//t, c, t, t)
    return tiles


def create_tiles(array_dir, tile_size=224, file_ext='tif'):

    dir_name = array_dir.split(os.path.sep)[-1]
    out_dir = array_dir.replace(dir_name, f'tile_{dir_name}')
    os.makedirs(out_dir, exist_ok=True)

    files = glob(os.path.join(array_dir, f'*.{file_ext}'))
    with ThreadPoolExecutor(max_workers=8) as exec:
        future_to_file = {exec.submit(create_tile, f_path, tile_size, file_ext): f_path
                          for f_path in files}

        pbar = tqdm(as_completed(future_to_file), total=len(future_to_file))
        for future in pbar:
            try:
                tiles = future.result()
            except Exception as e:
                raise e

            f_path = future_to_file[future]
            for i, tile in enumerate(tiles):
                tile = np.squeeze(tile, axis=0)  # (3, h, w) or (h, w)

                f_name = f_path.split(os.path.sep)[-1]
                components = f_name.split('.')
                components.insert(-1, f't{i}')

                out_f_path = f_path.replace(f_name, '.'.join(components))
                out_f_path.replace('.tif', '.png')

                if len(tile.shape) == 3:
                    tile = tile.transpose((1,2,0))
                im = Image.fromarray(tile, mode='L' if len(tile.shape) == 2 else 'RGB')
                im.save(out_f_path)


def f_name_sort_key(f_name):
    dirs = f_name.split(os.path.sep)  # ['a, 'b', 'n.json']
    return int(dirs[-1].split('.')[0])  # use int(n) as sort key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run preprocessing for QFabric')
    parser.add_argument('--do', choices=['jsons', 'change_type_masks', 'csv'], type=str, default='jsons',
                        help='Which functionality to perform')
    parser.add_argument('--data_path', default='./QFabric', type=str, help='Root dir of QFabric dataset')
    parser.add_argument('--gjson_dir', default='./QFabric/QFabric_Labels/geojsons')
    parser.add_argument('--json_dir', default='./QFabric/QFabric_Labels/jsons')
    parser.add_argument('--raster_dir', default='./QFabric/rasters/')

    args = parser.parse_args()

    if args.do == 'jsons':
        print('Annotating jsons with geojson data.')
        annotate_jsons(args.gjson_dir, args.json_dir)
    elif args.do == 'change_type_masks':
        print('Creating change type masks for each location (1 per location)')
        create_change_type_masks(args.raster_dir, args.json_dir, QFabricDataset.CHANGE_TYPES)
    elif args.do == 'tile':
        print('Tiling change-type masks and rasters to smaller arrays')

        pass
    elif args.do == 'csv':
        print('Creating train-val-test csv files')

        pass
    else:
        raise NotImplementedError

    pass
