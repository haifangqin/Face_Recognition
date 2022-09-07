set -e
eval "$(conda shell.bash hook)"

NAME=face_rec
if ! conda env list|grep $NAME; then
    conda create -y -n $NAME python=3.7
fi
conda activate $NAME
# for face recognition and alignment
conda install -y opencv=3.4.2 
pip install opencv-python==4.6.0.66 Pillow==8.3.1 typing-extensions==3.10.0.0 requests==2.25.1 scikit-image==0.17.2 mxnet-neuron==1.5.1.1.6.1.0 scikit-learn==0.24.2
# for face quality
pip install numpy==1.17 loguru==0.6.0 tqdm==4.61.2


if ! ls|grep weights; then
    aws s3 cp s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/face_recognition/weights.zip .
    unzip weights.zip
fi

if ! ls|grep test_set; then
    aws s3 cp s3://hotstar-ads-ml-us-east-1-prod/content-intelligence/face_recognition/test_set.zip .
    unzip test_set.zip
fi
