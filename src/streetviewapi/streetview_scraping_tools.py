# -*- coding: utf-8 -*-
"""
Utilities to interact with the Google Street View Static API,
to perform metadata queries and download panoramas.

@author: Alberto Cattaneo
"""

import numpy as np
import random
import requests
from fastai.vision.core import PILImage
from typing import List
from shapely.geometry import Point, MultiPolygon

class StreetViewAPI(object):
  def __init__(self, api_key: str):
     self.meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?'
     self.meta_params = dict(key=api_key, source='outdoor')
     self.pic_base = 'https://maps.googleapis.com/maps/api/streetview?'
     self.pic_params = dict(key=api_key, fov=100)
     
  def query_pano_meta(self, point: tuple, radius: int = 2000) -> (tuple, str) or None:
    """
    Query metadata of a panorama close to a given location.
    point: target location (x=latitude, y=longitude)
    radius: radius (in meters) in which to search for a panorama, centered at target point.

    Returns None if no panorama is available.
    """
    params = self.meta_params
    params['radius'] = radius
    params['location'] = f"{point[0]},{point[1]}"
    meta_resp = requests.get(self.meta_base, params=params)
    infos = meta_resp.json()
    if meta_resp.ok and infos['status'] == 'OK':
      return (infos['location']['lat'], infos['location']['lng']), infos['pano_id']
    else:
      return None

  def query_image(self, pano: str, size: int, heading: int) -> PILImage:
    """
    Query Google Street View images from their panorama id.

    pano: panorama id
    size: size of the (square) picture
    heading: compass heading of the camera
    """
    params = self.pic_params
    params['pano'] = pano
    params['size'] = f"{size}x{size}"
    params['heading'] = heading
    pic_resp = requests.get(self.pic_base, params=params)
    return PILImage.create(pic_resp.content)


def sample_region(sv: StreetViewAPI, mpol: MultiPolygon, n_points: int, only_rand: bool =False) -> (List[Point], List[str]):
  """
  Build a grid of locations where Street View panoramas are available, inside a geographical region.
  sv: StreetViewAPI instance to retrieve metadata
  mpol: the MultiPolygon inside which the grid is constructed
  n_points: number of locations to generate
  only_rand: if True, all locations are chosen randomly. Otherwise, the alogithm first tries a regular grid of equidistant locations.
  """

  long_min, lat_min, long_max, lat_max = mpol.bounds
  grid = []
  pic_ids = []
  if not only_rand:
    sz = int(np.floor(np.sqrt(n_points)))
    xx, yy = np.meshgrid(np.linspace(long_min, long_max, sz), np.linspace(lat_min, lat_max, sz))
    
    #Regular grid across the region
    for x, y in zip(xx.flatten(), yy.flatten()):
      pnt = Point(x, y)
      if pnt.within(mpol):
        resp = sv.query_pano_meta((pnt.y, pnt.x))
        if resp:
          grid.append(Point(resp[0][1], resp[0][0]))
          pic_ids.append(resp[1])

  #If points are still missing (or only_rand=True), sample random locations
  while len(grid) < n_points:
    pnt = Point(random.uniform(long_min, long_max), random.uniform(lat_min, lat_max))
    if pnt.within(mpol):
      resp = sv.query_pano_meta((pnt.y, pnt.x))
      if resp:
        grid.append(Point(resp[0][1], resp[0][0]))
        pic_ids.append(resp[1])

  return grid, pic_ids