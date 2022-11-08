# -*- coding: utf-8 -*-
import clip
import torch
import numpy as np
from PIL import Image
from googletrans import Translator
from utils import get_device, get_model_and_preprocess

def encode(img=None, text=None):
    model,preprocess = get_model_and_preprocess()

    if text != None:
        input_text = clip.tokenize(text).to(get_device)
    elif img != None:
        image = Image.open(img)
        proceed_img = preprocess(image).unsqueeze(0).to(get_device)


    with torch.no_grad():
        if text != None:
            feat = model.encode_text(input_text)
        elif img != None:
            feat = model.encode_image(proceed_img)
        feat /= feat.norm(dim=-1, keepdim = True)
        feat = feat.cpu().numpy()

    return feat

def cos_sim(want, recipe_name):
    return np.dot(want, recipe_name) / (np.linalg.norm(want) * np.linalg.norm(recipe_name))


#encod cos_simを統合
def encode_cos(want,recipe_name):
    b = encode(text=recipe_name)

    return cos_sim(want,b[0])
    
#翻訳
def trans(text):
    translator = Translator()
    result = translator.translate(text, dest="en")

    return result.text

#配列作成
def insert(want, recipe_name, img_url, site_url):
    recipe_name_en = trans(recipe_name)
    cos_sim = encode_cos(want, recipe_name_en)
    
    recommend = []

    recommend.append(recipe_name)
    recommend.append(recipe_name_en)
    recommend.append(cos_sim)
    recommend.append(img_url)
    recommend.append(site_url)

    return recommend

#味に対する特徴量のリスト作成
def make_taste_list():
    taste_list = ['Plain meal', 'Light meal', 'Heavy meal', 'Healthy', 'Salty', 'Sweet', 'Spicy', 'Sour', 'Greasy']


    taste_encoded = []

    for i in range(len(taste_list)):
        temp = encode(taste_list[i])
        taste_encoded.append(temp[0])

    taste = {}

    for i in range(len(taste_list)):
        taste[taste_list[i]] = taste_encoded[i]

    return taste