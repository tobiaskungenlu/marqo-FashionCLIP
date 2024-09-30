MODEL_NAME=google/siglip-base-patch16-256-multilingual


for DATASET in 'deepfashion_inshop_chinese' 'atlas_chinese' 'deepfashion_multimodal_chinese' 'fashion200k_chinese' 'KAGL_chinese'  'polyvore_chinese'
do
    python3 eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name ${MODEL_NAME} \
        --run-name siglip
done