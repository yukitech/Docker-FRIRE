import clip
import torch
from PIL import Image
from utils import get_device, get_model_and_preprocess
import pickle

#味に対する特徴量のリスト作成
def make_taste_list():
    taste_list = ['Plain meal', 'Light meal', 'Heavy meal', 'Healthy', 'Salty', 'Sweet', 'Spicy', 'Sour', 'Greasy']


    taste_encoded = []

    for i in range(len(taste_list)):
      print(f"encode start {taste_list[i]}")
      temp = encode(None,taste_list[i])
      taste_encoded.append(temp[0])
      print("finish encode")

    taste = {}

    for i in range(len(taste_list)):
        taste[taste_list[i]] = taste_encoded[i]

    return taste

def encode(img, text):
    model,preprocess = get_model_and_preprocess()
    device = get_device()

    if text != None:
        input_text = clip.tokenize(text).to(device)
    elif img != None:
        image = Image.open(img)
        proceed_img = preprocess(image).unsqueeze(0).to(device)


    with torch.no_grad():
        if text != None:
            feat = model.encode_text(input_text)
        elif img != None:
            feat = model.encode_image(proceed_img)
        feat /= feat.norm(dim=-1, keepdim = True)
        feat = feat.cpu().numpy()

    return feat

if __name__ == '__main__':
  # taste_feats = make_taste_list()
  # with open('taste_feats.pkl', 'wb') as fw:
  #   pickle.dump(taste_feats, fw)
  with open('taste_feats.pkl', 'rb') as p:
    l = pickle.load(p)
    print(l)