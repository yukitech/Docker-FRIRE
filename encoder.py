# -*- coding: utf-8 -*-
import clip
import torch
import numpy as np
from PIL import Image
from googletrans import Translator
from utils import get_device, get_model_and_preprocess
import gc


def encode(img, text):
    device = get_device()

    if text != None:
        input_text = clip.tokenize(text).to(device)
    elif img != None:
        model,preprocess = get_model_and_preprocess()
        del model
        gc.collect()
        image = Image.open(img)
        proceed_img = preprocess(image).unsqueeze(0).to(device)

    model,preprocess = get_model_and_preprocess()
    with torch.no_grad():
        if text != None:
            feat = model.encode_text(input_text)
        elif img != None:
            feat = model.encode_image(proceed_img)
        
        del model
        gc.collect()
        feat /= feat.norm(dim=-1, keepdim = True)
        feat = feat.cpu().numpy()

    return feat


def cos_sim(want, recipe_name):
    return np.dot(want, recipe_name) / (np.linalg.norm(want) * np.linalg.norm(recipe_name))


#encod cos_simを統合
def encode_cos(want,recipe_name):
    recipe_name_feat = encode(None, recipe_name)

    return cos_sim(want,recipe_name_feat[0])
    
#翻訳
def trans(text):
    translator = Translator()
    result = translator.translate(text, dest="en")

    return result.text

#配列作成
def insert(value,want, recipe_name, img_url, site_url):
    recipe_name_en = trans(recipe_name)
    cos_sim = encode_cos(want, recipe_name_en)
    
    recommend = []

    recommend.append(value)
    recommend.append(recipe_name)
    recommend.append(recipe_name_en)
    recommend.append(cos_sim)
    recommend.append(img_url)
    recommend.append(site_url)

    return recommend