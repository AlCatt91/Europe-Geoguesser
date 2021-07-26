# -*- coding: utf-8 -*-


import torch
import numpy as np
import pandas as pd
import re
from pathlib import Path
from fastai.learner import Learner
from fastai.torch_core import show_image
from fastai.vision.core import PILImage
from fastcore.basics import fastuple

class ImageTuple(fastuple):
    """ Tuple of PILImages. """
    @classmethod
    def create(cls, fns): return cls(tuple(PILImage.create(f) for f in fns))
    
    def show(self, ctx=None, **kwargs): 
        t1,t2,t3 = self
        if not isinstance(t1, torch.Tensor) or not isinstance(t2, torch.Tensor) or not isinstance(t3, torch.Tensor) or t1.shape != t2.shape or t1.shape != t3.shape or t2.shape != t3.shape: return ctx
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2,line,t3], dim=2), ctx=ctx, **kwargs)


def distance_check(learn: Learner, final_grid: pd.DataFrame, all : bool = True) -> None:
  """
  Computes mean distance between the centroids of MultiPolygones in geometries and the predictions of learn, for elements in the validation dataloader.
  learn: model Learner
  final_grid: DataFrame containing MultiPolygons in 'geometry' column and list of locations (as Point) in 'grid_points' column
  all: if True, take as coordinates prediction the weighted average of all centroids; if False, consider only the centroid of the most likely MultiPolygon.
  """
  #Matrix of multipolygon centroids, indexed as the numerical labels in the Learner dataloaders
  centers = torch.tensor([[cell.centroid.x, cell.centroid.y] for cell in final_grid.loc[map(int, learn.dls.vocab), 'geometry']])
  
  squared_errors = []

  for path in learn.dls.valid.items:
    #Check whether the model takes as input a single image or an ImageTuple
    if not isinstance(path, Path):
      match = re.match(r".*_(\d+)_(\d+)_(\d+)$", path[0].stem).groups()
      img = ImageTuple.create(path[:-1])
    else:
      match = re.match(r".*_(\d+)_(\d+)_(\d+)$", path.stem).groups()
      img = PILImage.create(path)

    cell, img_id = match[0], int(match[1])
    pt = final_grid.loc[int(cell), 'grid_points'][img_id]
    #True coordinates of the location
    targ_coords = torch.tensor([pt.x, pt.y]) 

    _, _, probs = learn.predict(img)
    if all:
      #Prediction is weighted average of centroids
      pred_coords = probs @ centers
    else:
      #Prediction is the centroid of the most likely cell
      pred_coords = centers[int(torch.argmax(probs))]
    #Append to squared_errors the squared Euclidean distance between the true location and the predicted one 
    squared_errors.append(((targ_coords - pred_coords) ** 2).sum())

  print(f"Validation RMSE: {np.mean(np.sqrt(squared_errors)):.4f}")
  
  

def predict_coords(probs: torch.tensor, final_grid: pd.DataFrame, vocab: list) -> torch.tensor:
    """
    Computes latitude/longitude coordinates of the weighted average of MultiPolygon centroids.
    
    probs: tensor of weigths
    final_grid: dataframe with column 'geometry' containing the MultiPolygons
    vocab: the model's vocabulary (labels need to be the indices of final_grid)
    """
    centers = torch.tensor([[cell.centroid.x, cell.centroid.y] for cell in final_grid.loc[map(int, vocab), 'geometry']])
    coords = probs @ centers
    return coords