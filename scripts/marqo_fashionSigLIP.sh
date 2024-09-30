MODEL_NAME=Marqo/marqo-fashionSigLIP

# for DATASET in 'deepfashion_inshop_chinese' 'deepfashion_multimodal_chinese' 'fashion200k_chinese' 'KAGL_chinese' 'atlas_chinese' 'polyvore_chinese' #'iMaterialist'
for DATASET in 'deepfashion_inshop'
do
    python3 eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name ${MODEL_NAME} \
        --run-name Marqo-FashionSigLIP
done