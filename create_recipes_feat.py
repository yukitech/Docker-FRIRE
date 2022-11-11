from encoder import insert
from app import app
from app import Recipes
import pickle
import json
import os

def create_recipes_feat():
  taste_list = ['Plain meal', 'Light meal', 'Heavy meal', 'Healthy', 'Salty', 'Sweet', 'Spicy', 'Sour', 'Greasy']

  with app.app_context():
    recipe_items = Recipes.query.all()

  with open('frire_data/taste_feats.pkl', 'rb') as fr:
    taste = pickle.load(fr)

  if os.path.exists('frire_data/recipes_feat.json'):
    with open('frire_data/recipes_feat.json') as fr:
      recipes_feat_data = json.load(fr)
  else:
    recipes_feat_data = {}

  for recipe_item in recipe_items:
    results = []

    if recipe_item.recipeName in recipes_feat_data:
      print(f"{recipe_item.recipeName} is registered in the dictionary")
      continue

    print(recipe_item.recipeName)
    
    for value in taste_list:
      print(value)
      results.append(insert(value,taste[value], recipe_item.recipeName,recipe_item.recipeImg, 'url'))
    recipes_feat_data[recipe_item.recipeName] = create_dictionary_data(results)

  create_jsonfile(recipes_feat_data)



def create_dictionary_data(feat_results_data):
  value_dict = {}

  for data in feat_results_data:
    value_dict[data[0]] = data[3].item()

  return value_dict

def create_jsonfile(recipes_feat_data):
  with open('frire_data/recipes_feat.json', 'w') as fw:
    json.dump(recipes_feat_data, fw, ensure_ascii=False, indent=4)


if __name__ == '__main__':
  create_recipes_feat()