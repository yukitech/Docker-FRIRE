# -*- coding: utf-8 -*-
import clip
import torch

def get_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device

def get_model_and_preprocess():
  device = get_device()
  model,preprocess = clip.load("ViT-B/32", device = device)
  
  return model, preprocess