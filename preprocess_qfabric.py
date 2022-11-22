import argparse
import os
import ast
import json
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

from math import floor, ceil
from PIL import Image
from matplotlib.path import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from util.qfabric_dataset import QFabricDataset

HEIGHT = 9928
WIDTH = 9796


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


def annotate_jsons(gjson_path, json_path, num_workers=8):

    new_json_dir = json_path.replace('jsons', 'new_jsons')
    os.makedirs(new_json_dir, exist_ok=True)

    # Assume files are x.geojson and x.json, where x is some non-negative int
    gjson_files = glob(os.path.join(gjson_path, '*.geojson'))
    gjson_files = sorted(gjson_files, key=f_name_sort_key)

    json_files = glob(os.path.join(json_path, '*.json'))
    json_files = sorted(json_files, key=f_name_sort_key)

    with ThreadPoolExecutor(max_workers=num_workers) as exec:
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


def merge_coco_with_jsons(coco_dir, json_dir):
    out_dir = os.path.dirname(os.path.relpath(coco_dir))
    out_dir = os.path.join(out_dir, 'new_coco')

    json_files = glob(os.path.join(json_dir, '*.json'))
    json_files = sorted(json_files, key=f_name_sort_key)

    coco_files = glob(os.path.join(coco_dir, '*.json'))
    coco_files = [f for f in coco_files if not f.endswith('metadata.json')]
    loc_to_coco = {}
    for c_file in coco_files:
        with open(c_file, 'r') as f:
            coco = json.load(f)

        im_name = coco['images'][0]['name']
        loc = int(im_name.split('.')[0])
        loc_to_coco[loc] = c_file

    for i, j_file in tqdm(enumerate(json_files), total=len(json_files)):
        c_file = loc_to_coco[i]

        with open(j_file, 'r') as f:
            j_data = json.load(f)

        with open(c_file, 'r') as f:
            c_data = json.load(f)

        c_data['shapes'] = j_data['shapes']

        with open(os.path.join(out_dir, f'{i}.json')) as f:
            json.dump(c_data, f, indent=2)


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


def create_change_type_mask(json_file, change_types):
    # raster_file = raster_files[0]

    with open(json_file, 'r') as f:
        labels = json.load(f)

    h, w = HEIGHT, WIDTH

    # based on coco jsons
    raster_info = labels['images'][0]
    h, w = raster_info['height'], raster_info['width']

    # # based on old jsons
    # raster_file = os.path.split(raster_info['file_name'])[-1]
    # raster_file = os.path.join(raster_dir, raster_file)
    # with rasterio.open(raster_file) as raster:
    #     h = raster.height
    #     w = raster.width

    # all_coords = gen_grid_points(0, w, 0, h)  # (h*w, 2)
    # assert all_coords.shape == (h*w, 2)

    # # ASSUME: NO OVERLAP
    label_mask = np.zeros((h, w), dtype=np.uint8)
    # for shape in labels['annotations']:
    #     label_idx = shape['properties'][0]['labels'][0] + 1  # get change_type idx, + 1 to distinguish no_change
    for shape in labels['shapes']:
        label = shape['properties']['change_type']
        label_idx = change_types.index(label)

        poly_coords = np.array(shape['points']).reshape(-1, 2)  # (P, 2)

        intersection_mask, bounds = poly_mask(poly_coords, w, h)
        x_min, x_max, y_min, y_max = bounds

        fill_values = label_mask[y_min:y_max, x_min:x_max]
        f_h, f_w = fill_values.shape
        fill_values = fill_values.reshape(-1)
        fill_values[intersection_mask] = label_idx

        label_mask[y_min:y_max, x_min:x_max] = fill_values.reshape(f_h, f_w)

    return label_mask


def create_change_type_masks(json_dir, change_types, num_workers=8):
    dir_name = os.path.dirname(os.path.relpath(json_dir))
    out_dir = os.path.join(dir_name, 'change_type_masks')
    os.makedirs(out_dir, exist_ok=True)

    json_files = glob(os.path.join(json_dir, '*.json'))
    # json_files = [f for f in json_files if not f.endswith('metadata.json')]
    json_files = sorted(json_files, key=f_name_sort_key)

    # def raster_sort_key(f_path):
    #     f_name = f_path.split(os.path.sep)[-1]
    #     f_id, d, date_str, _ = f_name.split('.')
    #     return (int(f_id), d)
    # raster_files = glob(os.path.join(raster_dir, '*.tif'))
    # raster_files = sorted(raster_files, key=raster_sort_key)
    #
    # assert 5*len(json_files) == len(raster_files), \
    #     f"Mismatch in num labels {len(json_files)}, and rasters {len(raster_files)}"

    with ThreadPoolExecutor(max_workers=num_workers) as exec:
        future_to_mask = {
            exec.submit(
                create_change_type_mask,
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
            json_file = json_files[idx]
            f_name = os.path.split(json_file)[-1]

            mask = Image.fromarray(mask)
            mask.save(os.path.join(out_dir, f"{f_name.replace('.json', '')}.png"))


def create_tile(array_file, out_dir, tile_size=224, file_ext='tif'):
    if file_ext == 'tif':
        with rasterio.open(array_file) as rst:
            img = rst.read()  # (3, h, w)
            h, w = rst.height, rst.width
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

    # out file
    f_name = os.path.split(array_file)[-1]
    components = f_name.split('.')

    tile_dir = os.path.join(out_dir, str(components[0]))
    if len(components) > 3:
        # use date as dir as well
        tile_dir = os.path.join(tile_dir, '.'.join(components[1:3]))
    os.makedirs(tile_dir, exist_ok=True)
    for i, tile in enumerate(tiles):
        tile_components = [c for c in components]
        tile_components.insert(-1, f't{i}')
        out_f_name = '.'.join(tile_components).replace('.tif', '.png')
        out_f_path = os.path.join(tile_dir, out_f_name)

        if c == 3:
            tile = tile.transpose((1, 2, 0))  # (h, w, 3)
        else:
            tile = np.squeeze(tile, axis=0)  # (h, w)
        im = Image.fromarray(tile, mode='RGB' if c == 3 else 'L')
        im.save(out_f_path)


def create_tiles(array_dir, tile_size=224, file_ext='tif', num_workers=8):

    dir_name = os.path.dirname(array_dir).split(os.path.sep)[-1]
    out_dir = array_dir.replace(dir_name, f'tile_{dir_name}')
    os.makedirs(out_dir, exist_ok=True)

    files = glob(os.path.join(array_dir, f'*.{file_ext}'))
    with ThreadPoolExecutor(max_workers=num_workers) as exec:
        future_to_file = {exec.submit(create_tile, f_path, out_dir, tile_size, file_ext): f_path
                          for f_path in files}

        pbar = tqdm(as_completed(future_to_file), total=len(future_to_file))
        for future in pbar:
            exception = future.exception()
            if exception is not None:
                raise exception


def create_split_csv(split_file_ids, coco_json_files,
                     tile_raster_dir, tile_change_type_dir, tile_change_status_dir=None,
                     tile_size=224):
    data = {'image-id': [],
            'image:01': [], 'date:01': [], 'image-name:01': [],
            'image:02': [], 'date:02': [], 'image-name:02': [],
            'image:03': [], 'date:03': [], 'image-name:03': [],
            'image:04': [], 'date:04': [], 'image-name:04': [],
            'image:05': [], 'date:05': [], 'image-name:05': [],
            'change-type': [], 'change-type-name': [],
            'num-tiles': [],}
    for f_name in split_file_ids:
        f_path = coco_json_files[f_name]

        with open(f_path, 'r') as f:
            coco = json.load(f)

        data['image-id'].append(coco['info']['id'])

        images_info = coco['images']
        assert len(images_info) == 5

        for i in range(1, 6):
            im_info = images_info[i-1]
            f_name = im_info['name']
            components = f_name.split('.')
            assert len(components) == 3, f"{components} not in loc.dx.datetime format"
            loc = components[0]
            date_str = '.'.join(components[1:])

            tile_dir_path = os.path.join(tile_raster_dir, loc, date_str)
            data[f'image:0{i}'].append(os.path.abspath(tile_dir_path))
            data[f'image-name:0{i}'].append(f_name)

            date = im_info['date_captured']
            data[f'date:0{i}'].append(date)

        loc = images_info[0]['name'].split('.')[0]
        change_type_dir = os.path.join(tile_change_type_dir, loc)
        data['change-type'].append(os.path.abspath(change_type_dir))
        data['change-type-name'].append(f"{loc}.png")

        h, w = images_info[0]['height'], images_info[0]['width']
        h_t, w_t = h // tile_size, w // tile_size
        data['num-tiles'].append(h_t * w_t)

    df = pd.DataFrame(data=data)
    return df


def create_dataset_csv(coco_json_dir, tile_raster_dir, tile_change_type_dir, tile_change_status_dir=None,
                       tile_size=224):
    json_files = glob(os.path.join(coco_json_dir, '*.json'))
    metadata_file = [f for f in json_files if f.endswith('metadata.json')][0]
    json_files = [f for f in json_files if not f.endswith('metadata.json')]

    def f_name(f):
        return os.path.split(f)[-1]

    json_files = {f_name(f): f for f in json_files}

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    parent_dir = os.path.dirname(os.path.relpath(coco_json_dir))

    train_file_ids = metadata['dataset']['train']
    train_df = create_split_csv(
        train_file_ids, json_files, tile_raster_dir, tile_change_type_dir, tile_change_status_dir,
        tile_size=tile_size,
    )
    train_df.to_csv(os.path.join(parent_dir, 'train.csv'))


    val_file_ids = metadata['dataset']['val']
    val_df = create_split_csv(
        val_file_ids, json_files, tile_raster_dir, tile_change_type_dir, tile_change_status_dir,
        tile_size=tile_size,
    )
    val_df.to_csv(os.path.join(parent_dir, 'val.csv'))


def f_name_sort_key(f_name):
    dirs = f_name.split(os.path.sep)  # ['a, 'b', 'n.json']
    return int(dirs[-1].split('.')[0])  # use int(n) as sort key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run preprocessing for QFabric')
    parser.add_argument('--do', choices=['jsons', 'coco', 'change_type_masks', 'csv', 'tile'], type=str, default='jsons',
                        help='Which functionality to perform')
    parser.add_argument('--data_path', default='./QFabric', type=str, help='Root dir of QFabric dataset')
    parser.add_argument('--gjson_dir', default='./QFabric/QFabric_Labels/geojsons')
    parser.add_argument('--json_dir', default='./QFabric/QFabric_Labels/jsons')
    parser.add_argument('--coco_dir', default='./QFabric/random-split/COCO')
    parser.add_argument('--csv', default='./random-split1_2022_11_09-18_39_08/CSV/train.csv')
    parser.add_argument('--raster_dir', default='./QFabric/rasters/')
    parser.add_argument('--mask_dir', default='./QFabric/labels/change_type_masks')
    parser.add_argument('--tile_size', default=256, type=int)
    parser.add_argument('--num_workers', default=8, type=int)

    args = parser.parse_args()

    if args.do == 'jsons':
        print('Annotating jsons with geojson data.')
        annotate_jsons(args.gjson_dir, args.json_dir, args.num_workers)
    elif args.do == 'coco':
        print('Adding correct shapes info to coco files')
        merge_coco_with_jsons(args.coco_dir, args.json_dir)
    elif args.do == 'change_type_masks':
        print('Creating change type masks for each location (1 per location)')
        create_change_type_masks(args.json_dir, QFabricDataset.CHANGE_TYPES, args.num_workers)
    elif args.do == 'tile':
        print('Tiling rasters to smaller arrays')
        create_tiles(args.raster_dir, tile_size=args.tile_size, file_ext='tif', num_workers=args.num_workers)
        print('Tiling change type masks to smaller arrays')
        create_tiles(args.mask_dir, tile_size=args.tile_size, file_ext='png', num_workers=args.num_workers)
        pass
    elif args.do == 'csv':
        print('Creating train-val-test csv files')
        create_dataset_csv(args.coco_dir, args.raster_dir, args.mask_dir)
        pass
    else:
        raise NotImplementedError

    pass
