import cv2
import time
import FaceNet
import argparse
import os
import numpy as np
from PIL import Image
import json

def save_bbox_json(faces, dicts, img):
    '''
    save bbox to json
    '''
    for face in faces:
        temp = {'img': img, 'bbox':[str(face.bounding_box[0]), str(face.bounding_box[1]), str(face.bounding_box[2]), str(face.bounding_box[3])], 'score': str(face.score), 'id': str(face.id)}
        dicts.append(temp)
    return dicts

def save_bbox_json_rec(faces, dicts, name=None, img=None):
    '''
    save bbox to json with identity name
    '''
    for face in faces:
        temp = {'img':img, 'identity': face.name, 'bbox':[str(face.bounding_box[0]), str(face.bounding_box[1]), str(face.bounding_box[2]), str(face.bounding_box[3])], 'score': str(face.score), 'id': str(face.id)}
        dicts.append(temp)
    return dicts


def Infor_annotation(frame, faces):
    '''
    Add recognition results to image
    '''
    if faces is not None and len(faces) > 0:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (255, 0, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[1]+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                            thickness=2, lineType=2)

    cv2.putText(frame, "25 fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def main(arg):

    if not os.path.isdir(arg.save_root_path):
        os.mkdir(arg.save_root_path)
    # for saving the results images with bbox and identity
    if not os.path.isdir(os.path.join(arg.save_root_path, 'detect_imgs')):
        os.mkdir(os.path.join(arg.save_root_path, 'detect_imgs'))
        img_save_path = os.path.join(arg.save_root_path, 'detect_imgs')
    if arg.mode == 'recognition' or arg.mode == 'make_features':
        # for saving the aligned faces images
        if not os.path.isdir(os.path.join(arg.save_root_path, 'aligned_faces')):  
            os.mkdir(os.path.join(arg.save_root_path, 'aligned_faces'))
            faces_save_path = os.path.join(arg.save_root_path, 'aligned_faces')
        # for saving the features in .npy format
        if not os.path.isdir(os.path.join(arg.save_root_path, 'features_npy')):  
            os.mkdir(os.path.join(arg.save_root_path, 'features_npy'))
            npy_save_path = os.path.join(arg.save_root_path, 'features_npy')
    
    
    # list all the images to recognition 
    frame_list = os.listdir(arg.target_rec_path)
    # initialize a recognition object
    face_recognition = FaceNet.Recognition(arg.gallery_path)               
    
    # save bbox to json
    if not arg.bbox_save_json == '':
        dicts = []

    for frame_name in frame_list:
        start_time = time.time()
        frame = cv2.imread(os.path.join(arg.target_rec_path, frame_name))
        if frame is None:
            print('Read failed in {}'.format(frame_name))
            continue

        if arg.mode == 'recognition':
            print('Recognize {} now...'.format(frame_name))
            faces = face_recognition.identify(frame)
        elif arg.mode == 'detection':
            print('Only detection {} faces now...'.format(frame_name))
            faces = face_recognition.detect.find_faces(frame)
        elif arg.mode == 'make_features':
            print('Making features {} now...'.format(frame_name))
            faces = face_recognition.detect.find_faces(frame)
            for i, face in enumerate(faces):
                face.embedding = face_recognition.encoder.generate_embedding(face)
        
        if not arg.bbox_save_json == '':
            if arg.mode == 'recognition':
                dicts = save_bbox_json_rec(faces, dicts, None, frame_name)
            else:
                dicts = save_bbox_json(faces, dicts, frame_name)
        
        # save all the aligned faces and embeddings
        for i,face in enumerate(faces):
            new_image = np.transpose(face.image, (1, 2, 0))[:, :, ::-1]
            out = Image.fromarray(new_image)
            out = out.resize((112, 112))
            out = np.asarray(out)
            # cv2.imshow('crop',out)
            if not npy_save_path == '':      
                np.save('%s/%s#%s#id_%s.npy'%(npy_save_path,frame_name[:-4], face.name,face.id),face.embedding)
            if not faces_save_path == '':      
                cv2.imwrite('%s/%s#%s#id_%s.jpg'%(faces_save_path, frame_name[:-4], face.name, face.id),out)
        # Check our current fps
        end_time = time.time()
        print(end_time - start_time)
        Infor_annotation(frame, faces)
        save_name = os.path.join(img_save_path, frame_name)
        cv2.imwrite(save_name, frame) 
    # write bbox results to json
    if not arg.bbox_save_json == '':
        out_file = open(os.path.join(arg.save_root_path, arg.bbox_save_json), "w")
        json.dump(dicts, out_file, indent = 4)
        out_file.close()         
    # generate img.list for face quality
    import glob
    src_path = os.path.abspath(faces_save_path)
    src = glob.glob(os.path.join(src_path, '*.jpg'))
    with open(os.path.join(arg.save_root_path, 'img.list'), 'w') as fout:
        for fi in src:
            fout.write('{}\n'.format(fi))


if __name__ == "__main__":
    parses = argparse.ArgumentParser(description="get the name and embedding,identity of faces")
    parses.add_argument('--target_rec_path',type = str,default='./test_set',help='the directory of images for recognition')
    parses.add_argument('--save_root_path',type = str,default='./results',help='the directory for saving the recognition images')
    parses.add_argument('--gallery_path', type = str, default='', help='the path to the gallery features')
    parses.add_argument('--bbox_save_json',type = str,default='test.json',help='the json file name for bbox saving')
    parses.add_argument('--mode',type = str,default='recognition',choices=['recognition', 'detection', 'make_features'], \
        help='for recognition mode, will generate all the materials. for detection mode, only the detection model will run. for make_features, will get the embedding and faces.')
    args = parses.parse_args()
    main(args)
