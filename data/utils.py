from datasets import load_dataset
import logging
import os
from PIL import Image
from io import BytesIO
import numpy as np
import sys
import json
import ast

token="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"

def copy_image(example, source_images):
    example['image'] = source_images.pop(0)
    return example

def filter_image_channels(example):
    # Assuming images are stored as a NumPy array or PIL Image
    image = example['image']
    test = np.array(image)
    
    # Check if the image has more than 2 channels
    # This assumes the image is a 3D array (height, width, channels)
    if len(test.shape) == 3:
        return True
    else:
        print(test.shape)
        return False

class Transform(object):
    def __init__(self, tokenizer, preprocess, doc_text_cols):
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.doc_text_cols = doc_text_cols
    def __call__(self, batch):
        if 'image' in batch:
           # print(batch['image'][0])
           #  sys.exit()
            if isinstance(batch['image'][0], dict):
                batch['image'] = [Image.open(BytesIO(img['bytes'])) for img in batch['image']]
                batch['image'] = [self.preprocess(img) for img in batch['image']]
            elif isinstance(batch['image'][0], str):
                batch['image'] = [ast.literal_eval(img) for img in batch['image']]
                batch['image'] = [self.preprocess(Image.open(BytesIO(img['bytes']))) for img in batch['image']]
            else:
                # for i in range(len(batch['image'])):
                #     img = np.array(batch['image'][i])
                #     print(img.shape)
                #     if len(img) == 2:
                #         print(img.shape)
                #         img = np.expand_dims(img, axis=-1)
                #     img = self.preprocess(img)
                #     batch['image'][i] = img
                    
                batch['image'] = [self.preprocess(img) for img in batch['image']]
            
        if self.doc_text_cols:
            for col in self.doc_text_cols:
                batch[col] = [self.tokenizer(text)[0] for text in batch[col]]
        return batch

def get_dataset(args, tokenizer, preprocess):
    print(args.dataset_config["hf_dataset"])
    print()
    logging.info('Loading dataset from huggingface.')
    doc_dataset = load_dataset(args.dataset_config["hf_dataset"], num_proc=os.cpu_count(), cache_dir=args.cache_dir, token=token)['data']
    # img_dataset = load_dataset(args.dataset_config["hf_dataset"].split("_")[0], num_proc=os.cpu_count(), cache_dir=args.cache_dir)['data']
    # print(doc_dataset[0])
    # source_images = [example['image'] for example in img_dataset]
    # print(source_images[0])
    # doc_dataset.remove_columns(['image'])
   #  doc_dataset = doc_dataset.map(lambda example: copy_image(example, source_images))
    # doc_dataset = doc_dataset.filter(filter_image_channels)
    # del img_dataset
    # doc_dataset.add_column('image', source_images)
    
    doc_text_cols, query_text_cols = set(), set()
    for task in args.dataset_config["tasks"]:
        # Document columns
        for doc_col in task["doc_col"]:
            if doc_col != 'image':
                doc_text_cols.add(doc_col)
        # Query columns
        for query_col in task["query_col"]:
            query_text_cols.add(query_col)
    item_ID = [str(id) for id in doc_dataset.data['item_ID'].to_pylist()]
    doc_dataset = doc_dataset.remove_columns([col for col in doc_dataset.column_names if col!='image' and (col not in doc_text_cols)])
            
    # Apply transform
    transform = Transform(tokenizer, preprocess, list(doc_text_cols))
    doc_dataset.set_transform(transform)
        
    return doc_dataset, item_ID
