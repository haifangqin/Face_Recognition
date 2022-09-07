# Face_Recognition
This repository includes face detection, face alignment, face clustering, face quality, face recognition


## Wiki 
Project page: 
[Face Recognition in videos](https://hotstar.atlassian.net/wiki/spaces/HP2/pages/3565977601/Face+Recognition+in+Videos)

## Quick Start
* Tested only on Linux machine
```bash
# on ec2 instance, auto-download models from S3
bash setup.sh

# run inference
#     1. face detection 
#     2. face alignment for face quality and face recognition
#     3. face embedding extraction
bash infer.sh $KEYF_DIR
```

## Pre-trained model 
* AWS S3 Link: ```s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/face_recognition/weights.tar.gz```

Unzip the model 

More detailed results are in Wiki page.

## Data 
1. Auto download from setip.sh or download yourself[here](s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/face_recognition/test_set.tar.gz) 
    

## Output file
Inside results directory, there could be three types of output associated with each video. 
1. ```<SAVE_DIR>/aligned_faces``` - Aligned face images, size is 112x112, can be as face quality input directly.
2. ```<SAVE_DIR>/detect_imgs``` - Detection images results, original size, can used for check the detection results or recognition results. 
3. ```<SAVE_DIR>/features_npy``` - Extrated embeddings, size is 512, can be used for clustering and recognition.
4. ```<SAVE_DIR>/img.list``` - All the aligned face images path list, used for face quality inference.
5. ```<SAVE_DIR>/test.json``` - The detection results.
6. ```<SAVE_DIR>/feat.list``` - All the output from face quality inference, used for calculating face quality score.
7. ```<SAVE_DIR>/output.json``` - Extrated embeddings, size is 512, can be used for clustering and recognition.


## Artwork Clustering
This will cluster the faces according to the embeddings
```bash
python faceClustering.py --filtered_csv $FILTERED_CSV --ouput_csv $OUTPUT_CSV --output_cluster_imgs $OUTPUT_CLUSTER_IMGS
```
