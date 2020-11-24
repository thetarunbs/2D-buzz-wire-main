'''
Module Name : DatasetGenerator.py
Author	    : Tarun Bhargav Sriram
Organization: DEMCON Advanced Mechatronics, TU Eindhoven

Description : Creating a dataset for CNN feature estimation in the Buzz Wire Project
			  Dataset of images and labels in a new directory
			  Images are jpg
			  labels in a csv file, first element in a column - image name, 50 successive elements - labels
			  Images with different agent and wire configurations are created 
			  Wire function created new for each iteration of image creation
			  Agent rotated randomly in each iteration
			  User can save image created, skip saving and exit
			  Hard exit at 2000 images created

Non-standard packages used - cv2, numpy, pandas, matplotlib

Home directory - /DatasetGeneration - v5/
Locally built Module imports - ImageMaker (location: /DatasetGeneration - v3/ImageMaker.py)
'''

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ImageMaker
import os
import time

# Image size
X_LENGTH=256
Y_LENGTH=256

# Image attributes
image_size=(X_LENGTH,Y_LENGTH,3)

# Wire attributes
wire_n=5000
wire_x=np.linspace(0, wire_n, num=wire_n, endpoint=False)
wire_thickness=int(1*10)
wire_r=wire_thickness/2

# Agent attributes - diameter and thickness
agent_d=11*10
agent_l=1*10

# Encoding hyperparameters
n_enc_line=9
n_enc_circle=21
n_enc_wire=11
radius_future_wire=5*10

# agent params
x=0
y=0

#####
# Wrapping attributes
wire_attributes=dict(zip(['wire_x','wire_n','wire_r','wire_thickness'],[wire_x,wire_n,wire_r,wire_thickness]))
agent_attributes=dict(zip(['agent_length','agent_diameter'],[agent_l,agent_d]))
image_attributes=dict(zip(['image_size'],[image_size]))
hyperparameters=dict(zip(['n_line_encode','n_agent_angle_encode','n_wire_angle_encode','rad_wire_future'],
						[n_enc_line,n_enc_circle,n_enc_wire,radius_future_wire]))

####################
####################
# Directory for storing dataset
directory='Datasets/Datasetdemo/training'
labels_file='/1. train_labels.csv'
DATAPOINTS=5000
k=1000 # Starting index of images and labels
####################
####################

# Number of features
N=2*n_enc_line+n_enc_circle+n_enc_wire

# Dataset directory creation
try:
	os.makedirs(directory)
	print('Directory created')
	df=pd.DataFrame({'Index':np.array([i for i in range(N)])})
	df.to_csv(directory+labels_file, index=False)
except FileExistsError:
	print('Directory Already exists')
finally:
	df=pd.read_csv(directory+labels_file)

def wire_maker(x):
	'''
	Function for creating wire - with an explicit function

	arguments: x - x axis values of the image frame
			   agent_pos - agent position in space

	returns: y - y axis values of the wire in image
	'''
	A=150+(300-150)*np.random.rand(1)[0]
	B=(-45+90*np.random.rand(1)[0])*np.pi/180
	amp=0+256*np.random.rand(1)[0]
	c=-2+4*np.random.rand(1)[0]
	y=c*x + amp*np.sin(x*np.pi/A + B)
	return y

def orientation_validity(xagent,wire_y,theta):
	#####
	# checking agent angle validity and modifying
	xagent1=xagent+50
	if xagent1>=wire_y.shape[0]:
		xagent1=wire_y.shape[0]-1
	wire_ori= -np.arctan((wire_y[xagent1]-wire_y[xagent])/(xagent1-xagent))
	diff=abs(abs(wire_ori)-abs(theta))
	angle_validity_threshold=np.radians(30)
	angle_offset=np.radians(40)
	if diff<angle_validity_threshold or abs(diff-np.pi/2)<angle_validity_threshold:
		theta+=angle_offset
		#print('offset1 added')
	#####

	xagent1=xagent-50
	if xagent1<0:
		xagent1=0
	wire_ori= -np.arctan((wire_y[xagent1]-wire_y[xagent])/(xagent1-xagent))
	diff=abs(abs(wire_ori)-abs(theta))
	angle_validity_threshold=np.radians(30)
	angle_offset=np.radians(40)
	if diff<angle_validity_threshold or abs(diff-np.pi/2)<angle_validity_threshold:
		theta+=angle_offset
		#print('offset2 added')
	#####
	return theta

####################
####################
while True:

	##########
	# wire creation
	wire_y=wire_maker(wire_x)
	##########

	##########
	# agent pose 
	# theta in RADIANS
	theta=np.radians(360*np.random.rand(1)[0])
	#xagent=100 #for now
	xagent=np.random.choice(wire_x[128:-127],size=1)[0].astype(int)
	#xagent=4950#int(10+(50-10)*np.random.rand(1)[0])
	theta=orientation_validity(xagent,wire_y,theta)
	yagent=wire_y[xagent]
	agent_pos=np.array([xagent,
						yagent,
						theta])

	wire_y+=np.random.randint(low=-3*agent_d/4,high=3*agent_d/4)
	agent_attributes['agent_pos']=agent_pos
	wire_attributes['wire_y']=wire_y
	# Image and feature generation
	img,flag,features=ImageMaker.gridmaker(wire_attributes,agent_attributes,image_attributes,hyperparameters)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print('features: ',features.shape)
	#invalid condition
	if flag==True:
		continue
	'''
	print('\nline: ',features[:18].T)
	print('circle: ',features[18:39].T)
	print('wire: ',features[39:50].T,'\n')
	features=np.array(list(features.T)[0])
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	'''
	# Saving the image and labels OR skip saving the data OR exit
	k+=1
	filename='img_'+str(k)+'.jpg'
	cv2.imwrite(directory+'/'+filename,img)
	df[filename] = features
	df.to_csv(directory+labels_file, index=False)
	if k%10==0:
		print('k=',k)
	if k==DATAPOINTS:
		print('Reached ',k,' datapoints')
		print('Exiting')
		exit()
