import numpy as np
import json
import argparse
import os

def main(arg):
    feat_path = os.path.join(arg.save_root_path, 'feat.list')

    with open(feat_path, 'r') as f:
        lines = f.readlines()
    img_2_feats = {}
    img_2_mag = {}
    for line in lines:
        parts = line.strip().split(' ')
        imgname = parts[0]
        feats = [float(e) for e in parts[1:]]
        mag = np.linalg.norm(feats)
        img_2_feats[imgname] = feats/mag
        img_2_mag[imgname] = mag
    imgnames = list(img_2_mag.keys())
    mags = [img_2_mag[imgname] for imgname in imgnames]
    sort_idx = np.argsort(mags)


    f= open(os.path.join(arg.save_root_path, arg.bbox_save_json))
    data = json.load(f)
    '''
    {'img': '1260107559/shot_0020_img_1.jpg',
    'bbox': ['607', '190', '1021', '759'],
    'score': '0.9999654',
    'id': '1'}
    '''

    dicts = {}
    for idx in sort_idx:
        imgname = imgnames[idx].split('/')[-1]
        key_ = imgname[:-4]
        dicts[key_] = mags[idx]

    new_data = []
    for item in data:
        key_ = item['img'][:-4] + '#None#id_' + item['id']
        item['face_quality'] = dicts[key_]
        new_data.append(item)
    out_file = open(os.path.join(arg.save_root_path, arg.output_json), 'w')
    json.dump(new_data, out_file, indent=4)
    out_file.close()



if __name__ == "__main__":
    parses = argparse.ArgumentParser(description="add face quality score to the json")
    parses.add_argument('--save_root_path', type = str, default='./results', help='the path saving features list')
    parses.add_argument('--bbox_save_json',type = str,default='test.json',help='the json file name for bbox saving')
    parses.add_argument('--output_json',type = str,default='output.json',help='the json file name for bbox saving')
    args = parses.parse_args()
    main(args)


