set -e
eval "$(conda shell.bash hook)"

NAME=face_rec
conda activate $NAME

KEYF_DIR=$1
SAVE_DIR=results
# Run face detection, alignment, embedding extraxtion for KEYF_DIR, the results will save to SAVE_DIR
python faceRec_infer.py --target_rec_path $KEYF_DIR --save_root_path $SAVE_DIR --mode make_features

# Extract the face quality to feat.list
python MagFace/inference/gen_feat.py --inf_list $SAVE_DIR/img.list --feat_list $SAVE_DIR/feat.list --resume MagFace/checkpoints/magface_epoch_00025.pth

# Add face quality to json 
python addFaceQuality.py --save_root_path $SAVE_DIR
# Conduct the clustering
# python faceClustering_infer.py --feat_list $SAVE_DIR/feat.list --json_file $SAVE_DIR/test.json --output_file $SAVE_DIR/output.json
