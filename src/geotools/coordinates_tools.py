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


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def distance_check(learn: Learner, final_grid: pd.DataFrame, all : bool = True) -> None:
  """
  Computes mean distance between the locations in the validation dataloader of learn and the predictions of the model.
  learn: model Learner
  final_grid: DataFrame containing MultiPolygons in 'geometry' column and list of locations (as List[Point]) in 'grid_points' column
  all: if True, take as coordinates prediction the weighted average of all MultiPolygon centroids; if False, consider only the centroid of the most likely MultiPolygon.
  """
  #Matrix of multipolygon centroids, indexed as the numerical labels in the Learner dataloaders
  centers = torch.tensor([[cell.centroid.x, cell.centroid.y] for cell in final_grid.loc[map(int, learn.dls.vocab), 'geometry']])

  #Items of validation dataloader, divided into batches
  batch_items = chunks(learn.dls.valid.items, learn.dls.valid.bs)

  mean_batch_errors = []

  for itms, (x, y) in zip(batch_items, learn.dls.valid):
    #Check whether the model takes as input a single image or an ImageTuple
    if isinstance(itms[0], Path):
      #Extract infos on the location from path of item
      match = [re.match(r".*_(\d+)_(\d+)_", s.stem).groups() for s in itms]
    else:
      match = [re.match(r".*_(\d+)_(\d+)_", s[0].stem).groups() for s in itms]
    #Recover Points of locations from final_grid['grid_points']
    pts = [final_grid.loc[int(a[0]), 'grid_points'][int(a[1])] for a in match]
    #True coordinates of locations for images in the batch
    targ_coords = torch.tensor([(pt.x, pt.y) for pt in pts]).reshape(-1, 2)

    #Probabilities of cells
    probs, _ = learn.get_preds(dl=[(x,y)])
    if all:
      #Prediction is weighted average of centroids
      pred_coords = probs @ centers
    else:
      #Prediction is the centroid of the most likely cell
      pred_coords = centers[probs.argmax(-1)]
    #Append to mean_batch_errors the mean squared Euclidean distance between the true location and the predicted one 
    mean_batch_errors.append(((targ_coords - pred_coords) ** 2).sum(1).mean())

  print(f"Validation RMSE: {np.sqrt(np.mean(mean_batch_errors)):.4f}")
  

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