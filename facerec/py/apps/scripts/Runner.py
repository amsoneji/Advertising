import os
import sys
import time
import subprocess

from facedetect_modified import get_face
from test import produce_prediction

__MODEL_PATH = './model.pkl'
	

def generate_ad(prediction):
	ad_Process = ''
	if (prediction == 0):
		#female ad
		ad_Process = subprocess.Popen(["/usr/bin/qlmanage","-p", "/Users/aniketsoneji/Pictures/DSC_0098.JPG"])
	else:
		#male ad
		ad_Process = subprocess.Popen(["/usr/bin/qlmanage","-p","/Users/aniketsoneji/Pictures/DSC_0097.JPG"])

	print("Process id = %d\n") % (ad_Process.pid)
	return ad_Process


def close_ad(process):
	process.kill()

if __name__=='__main__':
	print 'Starting Gender Ads Runner'

	i = 0
	while (i < 5):
		face_path = get_face()
		prediction = produce_prediction(__MODEL_PATH, face_path)
		ad_Process = generate_ad(prediction)
		time.sleep(5)
		close_ad(ad_Process)
		i += 1

	print 'Ending Gender Ads Runner'
