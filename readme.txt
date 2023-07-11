

# Examples of execution

## Extract features from videos using I3D and train RTFM method 
python3 run.py \
--no-feature-extraction=False \
--make-list-file=False \
--segment-size=16 \
--feature-root-train-abnormal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/i3d_standarized/training/anomaly/ \
--feature-root-train-normal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/i3d_standarized/training/normal/ \
--feature-root-test-abnormal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/i3d_standarized/test/anomaly/ \
--feature-root-test-normal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/i3d_standarized/test/normal/ \
--test-labels=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/labels/test.txt \
--video-root-train-normal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/training/normal \
--video-root-test-normal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/test/normal \
--video-root-train-abnormal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/training/anomaly \
--video-root-test-abnormal=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/test/anomaly \
--re-run-test=False \
--root=/media/denis/dados/CamNuvem \
--make-hd5-file=False \
--crop-10=False \
--feature-extractor=i3d \
--replace=True \
--gpu-id=cuda:0 \
--feature-size=2048 \
--anomaly-detection-method=RTFM \
--dataset=camnuvem

## Training the RTFM videosurveillance anomaly detection using pre-extracted feature vectors AND NOT extract features
python3 run.py \
--no-feature-extraction=True \
--make-list-file=False \
--segment-size=16 \
--test-labels=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/labels/test.txt \
--re-run-test=False \
--root=/media/denis/dados/CamNuvem \
--make-hd5-file=False \
--crop-10=False \
--feature-extractor=i3d-standarized \
--gpu-id=cuda:0 \
--feature-size=2048 \
--anomaly-detection-method=RTFM \
--dataset=camnuvem

## Training the WSAL method for the first time. We have to create Hd5 file first. 
python3 run.py \
--no-feature-extraction=True \
--make-list-file=False \
--segment-size=16 \
--test-labels=/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/labels/test.txt \
--re-run-test=False \
--root=/media/denis/dados/CamNuvem \
--make-hd5-file=True \
--crop-10=False \
--feature-extractor=i3d-standarized \
--gpu-id=cuda:0 \
--feature-size=2048 \
--anomaly-detection-method=WSAL \
--dataset=camnuvem


Before continue, run
apt-get update && apt-get install -y python3-opencv

scp -r 