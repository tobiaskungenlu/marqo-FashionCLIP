MODEL_NAME=Marqo/marqo-fashionSigLIP

# for DATASET in 'deepfashion_inshop' 'deepfashion_multimodal' 'fashion200k' 'KAGL' 'atlas' 'polyvore' 'iMaterialist'
for DATASET in 'deepfashion_inshop'
do
    python3 eval.py \
        --dataset-config ./configs/${DATASET}.json \
        --model-name ${MODEL_NAME} \
        --run-name Marqo-FashionSigLIP
done