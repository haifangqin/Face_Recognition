
import cv2
import os 
from matplotlib import pyplot as plt
from glob import glob
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def show_multiple_imgs(imgs, col=10, save_name=None):
    '''
    imgs: n x imgs
    col: max col
    '''
    #imgs = []
    #for f in files:
    #    imgs.append(cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB))
    w, h, c = imgs[0].shape
    n = len(imgs)
    row = int(n / col) + 1 if n % col else int(n / col)
    if n % col:
        imgs += [np.zeros((w, h, c), dtype=int) for i in range(col - n % col)]
    imgs = np.concatenate([np.concatenate(imgs[i * col : (i + 1) * col], axis=1) for i in range(row)], axis=0)
    if save_name == None:
        plt.figure(figsize=(6*col, 5*row))
        plt.imshow(imgs)
    else:
        cv2.imwrite(save_name, imgs)

def filter_key(data):
    # filtered csv
    
    filtered_img_list = data.img.tolist()
    filtered_id_list = data.id.tolist()
    filtered_keys = []

    for i, img_name in enumerate(filtered_img_list):
        filtered_keys.append(img_name[:-4] + '#None#id_' + str(filtered_id_list[i]))
        #print(filtered_keys)
    return filtered_keys

def main(arg):
    # get filter key
    data = pd.read_csv(arg.filtered_csv)
    filtered_keys = filter_key(data)
    # check distance within cluster 
    img_root_path = os.path.join(arg.save_root_path, 'aligned_faces')
    root_path = os.path.join(arg.save_root_path, arg.output_cluster_imgs)
    key_list = []
    face_id_list = []
    cluster_num = []
    inter_cluster_score = []

    img_list = []
    src = sorted(glob(os.path.join(os.path.abspath(arg.save_root_path), 'features_npy', '*.npy')))
    X = []
    for f in src:
        if not f.split('/')[-1][:-4] in filtered_keys:
            continue
        fea = np.load(f)
        X.append(fea)
        img_list.append(f)
    print('video has {} features'.format(len(X)))
    #Y = TSNE(n_components=2, random_state=17).fit_transform(X)
    #plt.scatter(Y[:, 0], Y[:, 1]);

    silhouette_score = []
    for k in range(3, 21): # [3,20]
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        silhouette_score.append(metrics.silhouette_score(X, kmeans.labels_, metric='cosine'))
    # print(silhouette_score)
    select_k = silhouette_score.index(max(silhouette_score))
    print('select k = {}, silhouette_score is {}'.format(select_k+3, silhouette_score[select_k]))
    kmeans = KMeans(n_clusters=select_k+3, random_state=0).fit(X)
    Centers = []
    for i, cluster in enumerate(kmeans.labels_):
        npy_path = img_list[i]
        fea = np.load(npy_path)
        Centers.append(kmeans.cluster_centers_[cluster])
    temp_dis = cosine_distances(Centers, X)
    dis_list = {}
    for i, cluster in enumerate(kmeans.labels_):
        if not str(cluster) in dis_list:
            dis_list[str(cluster)] = []
        dis_list[str(cluster)].append(temp_dis[i,i])
        
    for key,value in dis_list.items():
        print('{}, len {} dist {}'.format(key, len(value), np.mean(value)))
    # copy the feature images to cluster directory
    if not os.path.isdir(root_path):
        os.mkdir(root_path)

    for i, cluster in enumerate(kmeans.labels_):
        if not os.path.isdir(os.path.join(root_path, str(cluster))):
            os.mkdir(os.path.join(root_path, str(cluster)))
        npy_path = img_list[i]
        img_path = npy_path.split('/')[-1][:-4] + '.jpg'
        img_path = img_root_path  + img_path
        img_name = img_path.split('/')[-1]
        os.system('cp {} {}/{}'.format(img_path, os.path.join(root_path, str(cluster)), img_name))
        # get key and faceid
        key_temp = npy_path.split('/')[-1].split('#')[0] + '.jpg'
        face_id = int(npy_path[:-4].split('_')[-1])
        key_list.append(key_temp)
        face_id_list.append(face_id)
        cluster_num.append(int(cluster))
        inter_cluster_score.append(np.mean(dis_list[str(cluster)]))
    # show or save the cluster images
    for cluster in range(0, select_k+3):
        show_path = os.path.join(root_path, str(cluster))
        img_show_list = os.listdir(show_path)
        show_imgs = []
        for show_f in img_show_list:
            show_imgs.append(cv2.imread(os.path.join(show_path, show_f)))
        show_multiple_imgs(show_imgs, col=6,save_name=os.path.join(root_path, str(cluster)+'_vis.jpg'))


    new_df = pd.DataFrame(list(zip(key_list, face_id_list, cluster_num, inter_cluster_score)), columns=['img', 'id', 'episode_cluster_id', 'intra_cluster_score'])
    new_data = pd.merge(data, new_df, on=['img', 'id'], how='left')
    new_data.to_csv(os.path.join(arg.save_root_path, arg.output_csv), index=False)
    # new_data.head()


if __name__ == "__main__":
    parses = argparse.ArgumentParser(description="get the name and embedding,identity of faces")
    parses.add_argument('--save_root_path',type = str,default='./results',help='the root directory for saving materials')
    parses.add_argument('--filtered_csv',type = str,default='',help='the filtered csv file')
    parses.add_argument('--output_csv', type = str, default='', help='the output csv file')
    parses.add_argument('--output_cluster_imgs', type = str, default='', help='the path for saving the clustering images')
    args = parses.parse_args()
    main(args)