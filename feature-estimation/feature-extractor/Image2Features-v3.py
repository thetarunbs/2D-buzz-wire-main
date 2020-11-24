import subprocess
import sys
import os

import tensorflow as tf

# Importing Pandas package
try:
	import pandas as pd
except ImportError:
	print('Installing Pandas Package')
	subprocess.check_call([sys.executable, "-m", "pip", "install", 'pandas'])
finally:
	import pandas as pd

import numpy as np

# Importing OpenCV
try:
	import cv2
except ImportError:
	print('Installing OpenCV Package')
	subprocess.check_call([sys.executable, "-m", "pip", "install", 'opencv-python'])
finally:
	import cv2

import matplotlib.pyplot as plt

import pydot

# Data extractor function
def extractor(directory,labels_file):
	labels_df=pd.read_csv(directory+labels_file)
	filenames=labels_df.columns.values[1:]
	examples=np.asarray([cv2.imread(directory+file) for file in filenames])
	labels=np.asarray([(labels_df[file].values).astype(int) for file in filenames])
	return examples,labels

# Results graphing
def graphical(x,y,val_y,metric,visdir):
	plt.figure()
	plt.plot(x,y,'r',label='training '+metric)
	plt.plot(x,val_y,'b',label='validation '+metric)
	plt.title('Training and Validation '+metric)
	plt.legend()
	plt.savefig(visdir+'/'+metric+'_graph.png',dpi=140,transparent=True)
####################
####################
directory='Datasets/Datasetv4c-long_agent'
visdir='Vis/VisualizerMult12-unequal-lossweights'
####################
####################
training_directory=directory+'/training/'
testing_directory=directory+'/testing/'
training_labels='1. train_labels.csv'
testing_labels='1. test_labels.csv'

train_images,train_labels=extractor(training_directory,training_labels)
print('Training data loaded')
test_images,test_labels=extractor(testing_directory,testing_labels)
print('Testing data loaded')

# Data info
BATCH_SIZE=50

# Network Structure
IMAGE_SIZE=(256,256,3)
OUTPUT_SIZE=[9,9,21,11]
# Optimization
LOSS='categorical_crossentropy'
OPTIM=tf.keras.optimizers.Adam()
EPOCHS=100

####################
####################
####################
# Model Definition
inputs=tf.keras.Input(shape=IMAGE_SIZE,name='img')
# Common layers
x=tf.keras.layers.Conv2D(16,5,activation='relu')(inputs)
x=tf.keras.layers.MaxPooling2D(3)(x)
x=tf.keras.layers.Dropout(rate=0.2)(x)
x=tf.keras.layers.Conv2D(32, 5, activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(3)(x)
x=tf.keras.layers.Dropout(rate=0.2)(x)
x=tf.keras.layers.Conv2D(32, 5, activation='relu')(x)
x=tf.keras.layers.MaxPooling2D(3)(x)
x=tf.keras.layers.Dropout(rate=0.2)(x)
common_filters_output=tf.keras.layers.Flatten()(x)

# Feature 1 branch
x=tf.keras.layers.Dense(40,activation='relu')(common_filters_output)
x=tf.keras.layers.Dense(20,activation='relu')(x)
feature_1_output=tf.keras.layers.Dense(OUTPUT_SIZE[0],activation='softmax',name='feature_1_output')(x)

# Feature 2 branch
x=tf.keras.layers.Dense(40,activation='relu')(common_filters_output)
x=tf.keras.layers.Dense(20,activation='relu')(x)
feature_2_output=tf.keras.layers.Dense(OUTPUT_SIZE[1],activation='softmax',name='feature_2_output')(x)

# Feature 3 branch
x=tf.keras.layers.Dense(20,activation='relu')(common_filters_output)
#x=tf.keras.layers.Dense(20,activation='relu')(x)
feature_3_output=tf.keras.layers.Dense(OUTPUT_SIZE[2],activation='softmax',name='feature_3_output')(x)

# Feature 4 branch
x=tf.keras.layers.Dense(20,activation='relu')(common_filters_output)
x=tf.keras.layers.Dense(20,activation='relu')(x)
feature_4_output=tf.keras.layers.Dense(OUTPUT_SIZE[3],activation='softmax',name='feature_4_output')(x)

model=tf.keras.models.Model(
						inputs=inputs,
						outputs=[feature_1_output,
								feature_2_output,
								feature_3_output,
								feature_4_output],
						name='MultiFeaturesNet')

####################
####################
####################

####################
####################
# Model Visuaization
model.summary()
tf.keras.utils.plot_model(model, 'MultiFeaturesNet.png',show_shapes=True)
####################
####################

model.compile(
		optimizer=OPTIM,
		loss={'feature_1_output':LOSS,
			'feature_2_output':LOSS,
			'feature_3_output':LOSS,
			'feature_4_output':LOSS,
			},
		loss_weights={'feature_1_output':9/50,
				'feature_2_output':9/50,
				'feature_3_output':21/50,
				'feature_4_output':11/50
			},
		metrics={'feature_1_output':'accuracy',
				'feature_2_output':'accuracy',
				'feature_3_output':'accuracy',
				'feature_4_output':'accuracy'
			}
	)

history=model.fit(train_images,
					{
						'feature_1_output':train_labels[:,0:9],
						'feature_2_output':train_labels[:,9:18],
						'feature_3_output':train_labels[:,18:39],
						'feature_4_output':train_labels[:,39:50],
					},
					epochs=EPOCHS,
					batch_size=BATCH_SIZE,
					shuffle=True,
					verbose=1,
					validation_data=(test_images,
										{
											'feature_1_output':test_labels[:,0:9],
											'feature_2_output':test_labels[:,9:18],
											'feature_3_output':test_labels[:,18:39],
											'feature_4_output':test_labels[:,39:50],
										}
									),
					validation_steps=1
					)

####################
####################
# Visualizing graphical results
feat_acc=[history.history['feature_'+str(i)+'_output_accuracy'] for i in range(1,5)]
val_feat_acc=[history.history['val_feature_'+str(i)+'_output_accuracy'] for i in range(1,5)]
feat_loss=[history.history['feature_'+str(i)+'_output_loss'] for i in range(1,5)]
val_feat_loss=[history.history['val_feature_'+str(i)+'_output_loss'] for i in range(1,5)]
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(len(feat_acc[0]))

#graphical(x,y,val_y,metric,visdir)
try:
	os.mkdir(visdir)
	print('Directory created: ',visdir)
except FileExistsError:
	print('Directory ',visdir,' Already exists')

# Accuracy graphs
for i in range(4):
	graphical(epochs,feat_acc[i],val_feat_acc[i],'accuracy_feature_'+str(i+1),visdir)
	graphical(epochs,feat_loss[i],val_feat_loss[i],'loss_feature_'+str(i+1),visdir)
graphical(epochs,loss,val_loss,'loss',visdir)

####################
####################
# Save Model
model.save(visdir+'/FENN.h5')


####################
####################
# Confusion Matrix
####################
####################