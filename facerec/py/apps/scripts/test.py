import sys, os
sys.path.append("../..")
# import facerec modules
from facerec.feature import Fisherfaces, SpatialHistogram, Identity
from facerec.distance import EuclideanDistance, ChiSquareDistance
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from facerec.validation import KFoldCrossValidation
from facerec.visual import subplot
from facerec.util import minmax_normalize
from facerec.serialization import save_model, load_model
# import numpy, matplotlib and logging
import numpy as np
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import matplotlib.cm as cm
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from facerec.lbp import LPQ, ExtendedLBP

def produce_prediction(model_path, image_path):
	# model_path = sys.argv[1]
	# image_path = sys.argv[2]

	model = load_model(model_path)
	im = Image.open(image_path)
	im = im.convert("L")
	# resize to given size (if given)
	sz = (300,300)
	if (sz is not None):
	    im = im.resize(sz, Image.ANTIALIAS)
	X = np.asarray(im, dtype=np.uint8)

	prediction = model.predict(X)[0]
	print("Prediction : %d") % (prediction)
	return prediction
