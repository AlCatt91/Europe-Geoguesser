### Neural networks for image geolocation in Europe

The [notebook](https://github.com/AlCatt91/Europe-Geoguesser/blob/main/Geoguesser.ipynb) addresses the problem of geolocating a landscape picture using computer vision Deep Learning algorithms. The task is framed as a classification problem: given a picture from a European country we aim to identify the correct region of provenance (the country where it was taken, or the specific cell in a finer grid).

Data scraping for model training and validation is performed with the [Street View Static API](https://developers.google.com/maps/documentation/streetview/overview?hl=it); we provide a separate light [module](https://github.com/AlCatt91/Europe-Geoguesser/tree/main/src/streetviewapi) to interface with it.

### Setup

`!pip install git+https://github.com/AlCatt91/Europe-Geoguesser.git`
