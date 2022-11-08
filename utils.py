# -*- coding: utf-8 -*-
import clip

def get_device():
  device = "cpu"
  return device

def get_model_and_preprocess():
  model,preprocess = clip.load("ViT-B/32", device = get_device())
  
  return model, preprocess