import cv2
import numpy as np
import matplotlib.pyplot as plt

import subprocess
import sys
import time
try:
	import imutils
except ImportError:
	print('Installing imutils Package')
	subprocess.check_call([sys.executable, "-m", "pip", "install", 'imutils'])
finally:
	import imutils

def gridmaker(wire_attributes,agent_attributes,image_attributes,hyperparameters):
	# extracting wire attributes
	global wire_x
	global wire_y
	global wire_r
	global agent_pos
	global agent_l
	global agent_d
	global image_size
	global n_enc_line
	global n_enc_angle
	global n_enc_wire
	global rad_future_wire
	wire_x=wire_attributes['wire_x']
	wire_y=wire_attributes['wire_y']
	wire_r=wire_attributes['wire_r']
	wire_thickness=wire_attributes['wire_thickness']

	#extracting agent attributes
	agent_pos=agent_attributes['agent_pos']
	#imutils.rotate uses degrees, but agent_pos is in radians
	
	agent_l=agent_attributes['agent_length']
	agent_d=agent_attributes['agent_diameter']

	#extracting image attributes
	image_size=image_attributes['image_size']
	
	#extracting hyperparameters
	n_enc_line=hyperparameters['n_line_encode']
	n_enc_angle=hyperparameters['n_agent_angle_encode']
	n_enc_wire=hyperparameters['n_wire_angle_encode']
	rad_future_wire=hyperparameters['rad_wire_future']

	'''
	####################
	####################
	####################
	####################

	####################
	# experimenting with drawing a global view
	####################

	globalframe=np.zeros((500,1400,3),dtype=np.uint8)
	local_wire_x=wire_x[0:1200]+100

	local_wire_y=-75+256+0.1*wire_x[0:1200]+128*np.sin(wire_x[0:1200]*np.pi/150)

	wire_points=np.array([[local_wire_x[i],local_wire_y[i]] for i in range(len(local_wire_x))])
	cv2.polylines(globalframe,[np.int32(wire_points)],isClosed=False,thickness=wire_thickness,color=(0,0,255))
	
	offset=(int(local_wire_x[0]) - agent_l, 
				int(local_wire_y[0] - agent_d/2))
	points=[[0,0],[0,agent_d],[2*agent_l,agent_d],[2*agent_l,0]]
	points=[[point[0]+offset[0],point[1]+offset[1]] for point in points]
	wall_points = np.array(points,dtype=np.int32)
	cv2.fillPoly(globalframe,[wall_points],(0,0,255))

	offset=(int(local_wire_x[-1]) - agent_l, 
				int(local_wire_y[-1] - agent_d/2))
	points=[[0,0],[0,agent_d],[2*agent_l,agent_d],[2*agent_l,0]]
	points=[[point[0]+offset[0],point[1]+offset[1]] for point in points]
	wall_points = np.array(points,dtype=np.int32)
	cv2.fillPoly(globalframe,[wall_points],(0,0,255))
	### global wire drawn

	#drawing agent in global frame
	agent_pos=np.array([local_wire_x[200],
						local_wire_y[200],
						np.radians(30)])
	print(agent_pos)
	offset=(
			int(agent_pos[0] - agent_l/2),
			int(agent_pos[1] - agent_d/2)
				)
	#offset=(400,165)
	c,s=np.cos(-agent_pos[2]),np.sin(-agent_pos[2])
	R=np.array([[c, -s],[s, c]])
	# top part of the agent
	points=np.array([[0,0],[0,agent_d/2],[agent_l,agent_d/2],[agent_l,0]])
	points=np.array([point-np.array([agent_l/2,agent_d/2]) for point in points])
	points=[np.dot(R,point) for point in points]
	points=np.array([point+np.array([agent_l/2,agent_d/2]) for point in points]).astype(int)
	points=points.tolist()
	points=[[point[0]+offset[0],point[1]+offset[1]] for point in points]
	top_points = np.array(points,dtype=np.int32)
	# bottom part of the agent
	points=np.array([[0,agent_d/2],[0,agent_d],[agent_l,agent_d],[agent_l,agent_d/2]])
	points=np.array([point-np.array([agent_l/2,agent_d/2]) for point in points])
	points=[np.dot(R,point) for point in points]
	points=np.array([point+np.array([agent_l/2,agent_d/2]) for point in points]).astype(int)
	points=points.tolist()
	points=[[point[0]+offset[0],point[1]+offset[1]] for point in points]
	bottom_points = np.array(points,dtype=np.int32)

	cv2.fillPoly(globalframe,[top_points],(255,0,0))
	cv2.fillPoly(globalframe,[bottom_points],(0,255,0))
	#globalframe=np.hstack((space,globalframe))
	#globalframe=np.hstack((globalframe,space))
	cv2.imshow('image',globalframe)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.imwrite('global_obs.png',globalframe)
	exit()
	####################
	####################
	####################
	####################
	'''

	####################
	# initialization
	validity_flag=False
	features=np.zeros((2*n_enc_line+n_enc_angle+n_enc_wire,1))
	grid=np.zeros(image_size,dtype=np.uint8)
	wireframe=np.zeros(image_size,dtype=np.uint8)
	agentframe=np.zeros(image_size,dtype=np.uint8)

	####################
	#drawing the agent
	####################
	#offset for the agent in local image - DO NOT CHANGE
	offset=(image_size[0]/2 - agent_l/2, image_size[1]/2 - agent_d/2)
	# top part of the agent
	points=[[0,0],[0,agent_d/2],[agent_l,agent_d/2],[agent_l,0]]
	points=[[point[0]+offset[0],point[1]+offset[1]] for point in points]
	top_points = np.array(points,dtype=np.int32)

	# bottom part of the agent
	points=[[0,agent_d/2],[0,agent_d],[agent_l,agent_d],[agent_l,agent_d/2]]
	points=[[point[0]+offset[0],point[1]+offset[1]] for point in points]
	bottom_points = np.array(points,dtype=np.int32)

	# drawing the agent halves
	cv2.fillPoly(agentframe,[top_points],(255,0,0))
	cv2.fillPoly(agentframe,[bottom_points],(0,255,0))

	# orienting the agent in given angle
	agentframe=imutils.rotate(agentframe,agent_pos[2]*180/np.pi)
	
	#extracting corner points from agent
	# order of points - back top
	#					front top
	#					back bottom
	#					front bottom
	agent_corners=np.array([top_points[0,:],top_points[-1,:],
								bottom_points[1,:],bottom_points[2,:]])
	agent_corners=[point-np.array([128,128]) for point in agent_corners]
	c,s=np.cos(-agent_pos[2]),np.sin(-agent_pos[2])
	R=np.array([[c, -s],[s, c]])
	agent_corners=[np.dot(R,point) for point in agent_corners]
	agent_corners=np.array([point+np.array([128,128]) for point in agent_corners]).astype(int)
	agent_corners=np.array([(point[1],point[0]) for point in agent_corners])

	####################
	# drawing the wire
	####################
	#extracting local observation wire points
	indices=np.where(np.logical_and(wire_x>(agent_pos[0]-image_size[1]/2), wire_x<=(agent_pos[0]+image_size[1]/2)))
	local_wire_x=wire_x[int(agent_pos[0]-image_size[1]/2):
						int(agent_pos[0]+image_size[1]/2)]-agent_pos[0]+image_size[1]/2
	local_wire_y=wire_y[int(agent_pos[0]-image_size[1]/2):
						int(agent_pos[0]+image_size[1]/2)]-agent_pos[1]+image_size[1]/2
	wire_points=np.array([[local_wire_x[i],local_wire_y[i]] for i in range(len(local_wire_x))])
	cv2.polylines(wireframe,[np.int32(wire_points)],isClosed=False,thickness=wire_thickness,color=(0,0,255))

	####################
	# drawing a wall - more like a circle though
	####################
	if agent_pos[0] < image_size[0]/2:
		print('need a wall')
		local_wire_x=wire_x[int(image_size[0]/2-agent_pos[0]):image_size[0]]
		local_wire_y=wire_y[local_wire_x.astype(int)]
		wire_points=np.array([[local_wire_x[i],local_wire_y[i]] for i in range(len(local_wire_x))])
		cv2.polylines(wireframe,[np.int32(wire_points)],isClosed=False,thickness=wire_thickness,color=(0,0,255))
		cv2.circle(wireframe,tuple(wire_points[0].astype(int)),20,(0,0,255),-1)

	if agent_pos[0] > wire_x[-1]-image_size[0]/2:
		print('need a wall')
		local_wire_x=wire_x[int(agent_pos[0]-image_size[0]/2):]
		local_wire_y=wire_y[local_wire_x.astype(int)]
		wire_points=np.array([[local_wire_x[i],local_wire_y[i]] for i in range(len(local_wire_x))])
		cv2.polylines(wireframe,[np.int32(wire_points)],isClosed=False,thickness=wire_thickness,color=(0,0,255))
		end_point=(int(wire_points[-1,0]-agent_pos[0]+image_size[0]/2),
				int(wire_points[-1,1]-agent_pos[1]+image_size[0]/2))
		cv2.circle(wireframe,end_point,20,(0,0,255),-1)

	'''
	####################
	# Validity checking
	####################
	#####
	# checking corner validity - whether a corner goes into the wire
	for point in agent_corners:
		if wireframe[int(point[0]),int(point[1]),2]==255:
			validity_flag=True
			return agentframe+wireframe,validity_flag,features
			break
	
	#####
	# checking intersection validity - whether the agent intersects with the wire
	k=0
	coords=np.where(np.logical_or(agentframe[:,:,0],agentframe[:,:,1]))
	for i in range(coords[0].shape[0]):
		if wireframe[coords[0][i],coords[1][i],2]==255:
			k+=1
	if k<70 and k>200:
		validity_flag=True
		return agentframe+wireframe,validity_flag,features
	#####

	#####
	# checking agent location validity - intersection condition
	# edge equations
	m1=(agent_corners[1,0]-agent_corners[3,0])/(agent_corners[1,1]-agent_corners[3,1])
	m2=(agent_corners[0,0]-agent_corners[2,0])/(agent_corners[0,1]-agent_corners[2,1])
	c1=agent_corners[1,0]-m1*agent_corners[1,1]
	c2=agent_corners[0,0]-m2*agent_corners[0,1]
	# creating lines for agent edges
	edge1x=np.linspace(agent_corners[1,1],agent_corners[3,1],100)
	edge2x=np.linspace(agent_corners[0,1],agent_corners[2,1],100)
	edge1y=m1*edge1x+c1
	edge2y=m1*edge2x+c2
	## Intersection find - from github.com/sukhbinder - intersection/intersect/intersect.py
	# intersection with forward edge
	x_edge1,y_edge1=intersection(edge1x,edge1y,local_wire_x,local_wire_y)
	if len(x_edge1)==0 or len(y_edge1)==0:
		validity_flag=True
		return agentframe+wireframe,validity_flag,features
	if len(x_edge1)>1 or len(y_edge1)>1:
		x_edge1,y_edge1=x_edge1[0],y_edge1[0]
	# intersection with backward edge
	x_edge2,y_edge2=intersection(edge2x,edge2y,local_wire_x,local_wire_y)
	if len(x_edge2)==0 or len(y_edge2)==0:
		validity_flag=True
		return agentframe+wireframe,validity_flag,features
	if len(x_edge2)>1 or len(y_edge2)>1:
		x_edge2,y_edge2=x_edge2[0],y_edge2[0]
	#####
	'''
	# full local observation
	grid=wireframe+agentframe
	# recolouring intersection region
	coords=np.where(((grid[:,:,0]==255)&(grid[:,:,2]==255))|((grid[:,:,1]==255)&(grid[:,:,2]==255)))
	for i in range(coords[0].shape[0]):
		grid[coords[0][i],coords[1][i],2]=0
	
	theta=360*np.random.rand(1)[0]
	grid=imutils.rotate(grid,theta)
	agent_pos[2]+=np.radians(theta)
	agent_pos[2]%=np.radians(360)
	if agent_pos[2]*180/np.pi>=270 and agent_pos[2]*180/np.pi<360:
		agent_pos[2]-=(2*np.pi)
	
	'''
	####################
	# Encoding Features
	####################
	line_enc=encodify_line(agent_corners,(x_edge1,y_edge1),(x_edge2,y_edge2),agent_d,n_enc_line)
	agent_angle_enc=encoder_angle(agent_pos[2],2*np.pi,n_enc_angle,offset=np.pi/2)
	future_point,future_angle_enc=encodify_future_wire(agent_pos,agent_corners,(x_edge1,y_edge1),(x_edge2,y_edge2),local_wire_x,local_wire_y,rad_future_wire,n_enc_wire)
	features=np.vstack((line_enc,agent_angle_enc,future_angle_enc))
	'''
	#####
	'''
	# visualizing the future angle
	cv2.circle(wireframe,(future_point[0],future_point[1]),3,(0,255,255),-1)
	cv2.circle(wireframe,(agent_corners[3,1],agent_corners[3,0]),3,(255,255,255),-1)
	cv2.line(wireframe,tuple(future_point),(int((agent_corners[1,1]+agent_corners[3,1])/2),int((agent_corners[1,0]+agent_corners[3,0])/2)),(255,255,255),thickness=2)
	cv2.line(wireframe,(int((agent_corners[1,1]+agent_corners[3,1])/2),int((agent_corners[1,0]+agent_corners[3,0])/2)),(agent_corners[3,1],agent_corners[3,0]),(255,255,255),thickness=2)
	x1=np.array((int(future_point[1]),int(future_point[0])))
	x2=np.array((agent_corners[3,0],agent_corners[3,1]))
	x3=np.array((int((agent_corners[1,0]+agent_corners[3,0])/2),int((agent_corners[1,1]+agent_corners[3,1])/2)))
	a=x1-x3
	b=x2-x3
	a=a/np.linalg.norm(a)
	b=b/np.linalg.norm(b)
	angle=np.arccos(np.dot(a,b))
	print('future: ',angle*180/np.pi)
	cv2.imshow('image',wireframe+agentframe)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	exit()
	'''
	#####

	return grid,validity_flag,features

def encodify_line(agent_corners,intersect_edge1,intersect_edge2,agent_size,n_enc):
	#back edge intersection distance
	back_dist=np.linalg.norm(np.array([agent_corners[0,1]-intersect_edge2[0],
								agent_corners[0,0]-intersect_edge2[1]]))
	back_enc=encoder_line(back_dist,agent_size,n_enc)
	#back edge intersection distance
	front_dist=np.linalg.norm(np.array([agent_corners[1,1]-intersect_edge1[0],
								agent_corners[1,0]-intersect_edge1[1]]))
	front_enc=encoder_line(back_dist,agent_size,n_enc)
	#stacked encoded line features
	line_enc=np.vstack((front_enc,back_enc))
	return line_enc

#encodfiy_angle same as encoder_angle wth offset=np.pi/2
#offset for future wire angle calculation is 0

def encodify_future_wire(agent_pos,agent_corners,intersect_edge1,intersect_edge2,local_wire_x,local_wire_y,rad_future_wire,n_enc_wire):
	front_point=np.array([int(intersect_edge1[0]),int(intersect_edge1[1])])
	forward_points=np.tile(front_point,(local_wire_x.shape[0],1))
	wire_points=np.array([[local_wire_x[i],local_wire_y[i]] for i in range(len(local_wire_x))])
	dists=np.linalg.norm(wire_points-forward_points,ord=2,axis=1)
	indices=np.where(dists<rad_future_wire)[0]
	possible_future_points=np.array([[int(local_wire_x[indices[0]]),int(local_wire_y[indices[0]])],
									[int(local_wire_x[indices[-1]]),int(local_wire_y[indices[-1]])]])
	back_point=np.array([int(intersect_edge2[0]),int(intersect_edge2[1])])
	if np.linalg.norm(possible_future_points[0,:]-back_point)<np.linalg.norm(possible_future_points[1,:]-back_point):
		future_point=possible_future_points[1,:]
	else:
		future_point=possible_future_points[0,:]

	#points - future point, agent bottom front corner, agent front center
	x1=np.array((int(future_point[1]),int(future_point[0])))
	x2=np.array((agent_corners[3,0],agent_corners[3,1]))
	x3=np.array((int((agent_corners[1,0]+agent_corners[3,0])/2),int((agent_corners[1,1]+agent_corners[3,1])/2)))
	#unit vectors for lines made by future point to agent center, and agent center to agent front bottom corner
	a=(x1-x3)/np.linalg.norm(x1-x3)
	b=(x2-x3)/np.linalg.norm(x2-x3)
	#future angle required - using cosine rule
	future_angle=np.arccos(np.dot(a,b))
	future_wire_enc=encoder_angle(future_angle,np.pi,n_enc_wire,offset=0)
	return future_point,future_wire_enc

####################
def encoder_line(dist,normalizer,n):
	c=int(np.floor((dist/normalizer)*n))
	enc=np.zeros((n,1))
	enc[c]=1
	return enc

def encoder_angle(angle,normalizer,n,offset=np.pi/2):
	angle_norm=(angle+offset)/normalizer
	encoded_angle=int(np.floor(angle_norm*n))
	enc=np.zeros((n,1))
	enc[encoded_angle]=1
	return enc
####################

####################
####################
## Intersection find - from github.com/sukhbinder - intersection/intersect/intersect.py
def rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4

def rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj

def intersection(x1, y1, x2, y2):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]
####################
####################

# Global variables
####################
# wire attributes
wire_x=None
wire_y=None
wire_r=None
# agent attributes
agent_pos=None
agent_l=None
agent_d=None
#image attributes
image_size=None
#hyperparameters
n_enc_line=None
n_enc_angle=None
n_enc_wire=None
rad_future_wire=None

'''
##########
# experiments future wire
front_point=np.array([int(x_edge1),int(y_edge1)])
#point=(agent_corners[1,:]+agent_corners[3,:])/2
forward_points=np.tile(front_point,(local_wire_x.shape[0],1))
dists=np.linalg.norm(wire_points-forward_points,ord=2,axis=1)
indices=np.where(dists<rad_future_wire)[0]
possible_future_points=np.array([[int(local_wire_x[indices[0]]),int(local_wire_y[indices[0]])],
								[int(local_wire_x[indices[-1]]),int(local_wire_y[indices[-1]])]])
back_point=np.array([int(x_edge2),int(y_edge2)])
if np.linalg.norm(possible_future_points[0,:]-back_point)<np.linalg.norm(possible_future_points[1,:]-back_point):
	future_point=possible_future_points[1,:]
else:
	future_point=possible_future_points[0,:]
c,s=np.cos(agent_pos[2]),np.sin(agent_pos[2])
r = np.array([[s, -c],[c, s]])
relative_point=future_point-agent_pos[:-1]
point_rotated=relative_point.dot(r)
future_angle=np.arctan(point_rotated[1]/point_rotated[0])
cv2.circle(wireframe,tuple(future_point),3,(0,255,255),-1)
cv2.imshow('image',wireframe+agentframe)
cv2.waitKey(0)
cv2.destroyAllWindows()
exit()
##########
'''