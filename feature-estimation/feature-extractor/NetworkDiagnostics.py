import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import cv2
import os
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix

def feature_map_vis(model,img,visdir):
	successive_outputs=[layer.output for layer in model.layers[1:]]
	visualizer=tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

	x=img.reshape((1,)+img.shape)
	successive_feature_maps=visualizer.predict(x)
	layer_names=[layer.name for layer in model.layers[1:]]
	try:
		os.mkdir(visdir+'/feature_maps')
		print('Directory created: ',visdir+'/feature_maps')
	except FileExistsError:
		print('Directory ',visdir+'/feature_maps',' Already exists')

	for layer_name, feature_map in zip(layer_names,successive_feature_maps):

		if len(feature_map.shape)==4:
			n_features=feature_map.shape[-1]
			size=feature_map.shape[1]
			displaygrid=np.zeros((size,size*n_features))
			for i in range(n_features):
				x=feature_map[0,:,:,i]
				x-=x.mean()
				x/=x.std()
				x*=64
				x+=128
				x=np.clip(x,0,255).astype('uint8')
				displaygrid[:, i*size:(i+1)*size]=x
			scale=20./n_features
			plt.subplots(figsize=(scale*n_features,scale))
			plt.title(layer_name)
			plt.grid(False)
			plt.imshow(displaygrid,aspect='auto',cmap='viridis')
			plt.savefig(visdir+'/feature_maps/'+str(layer_name)+'.png',dpi=140,transparent=False,bbox_inches='tight')
			plt.close()
	#plt.show()
	
def features_stacker(model,directory):
	stack=cv2.imread(directory+model.layers[1].name+'.png')
	filenames=[file[:-4] for file in os.listdir(directory)]
	layer_names=[layer.name for layer in model.layers[2:] if layer.name in filenames]
	for layer in layer_names:
		img=cv2.imread(directory+layer+'.png')
		img=np.hstack((img,255*np.ones((img.shape[0],stack.shape[1]-img.shape[1],3),dtype=np.uint8)))
		stack=np.vstack((stack,img))
	cv2.imwrite(directory+'stacked_feature_maps.png',stack)

def filters_vis(model,visdir):
	try:
		os.mkdir(visdir+'/filters')
		print('Directory created: ',visdir+'/filters')
	except FileExistsError:
		print('Directory ',visdir+'/filters',' Already exists')

	for layer in model.layers:
		if 'conv' not in layer.name:
			continue
		filters,biases=layer.get_weights()
		f_min,f_max=filters.min(),filters.max()
		filters=(filters-f_min)/(f_max-f_min)
		n_filters=filters.shape[-1]
		ix=1
		plt.figure()
		for i in range(n_filters):
			f=filters[:,:,:,i]
			for j in range(f.shape[-1]):
				ax=plt.subplot(n_filters,f.shape[-1],ix)
				ax.set_xticks([])
				ax.set_yticks([])
				plt.imshow(f[:,:,j],cmap='gray')
				ix+=1
		plt.suptitle(layer.name)
		plt.savefig(visdir+'/filters/'+str(layer.name)+'.png',dpi=140,transparent=False,bbox_inches='tight')
	plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.tight_layout()

def extractor(directory,labels_file):
	labels_df=pd.read_csv(directory+labels_file)
	filenames=labels_df.columns.values[1:]
	examples=np.asarray([cv2.imread(directory+file) for file in filenames])
	labels=np.asarray([(labels_df[file].values).astype(int) for file in filenames])
	return examples,labels

if __name__=='__main__':
	visdir=input('Visualizer Directory Name: ')
	#visdir='VisualizerMult2'
	visdir='Vis/'+visdir
	model=tf.keras.models.load_model(visdir+'/FENN.h5')
	print('model loaded')
	print('#'*40)
	directory=input('Dataset Directory Name: ')
	directory='Datasets/'+directory
	training_directory=directory+'/training/'
	test_img=cv2.imread(training_directory+'img_10.jpg')
	#'''
	feature_map_vis(model,test_img,visdir)
	features_stacker(model,visdir+'/feature_maps/')
	filters_vis(model,visdir)
	#'''
	
	testing_directory=directory+'/testing/'
	testing_labels='1. test_labels.csv'
	test_images,test_labels=extractor(testing_directory,testing_labels)

	BATCH_SIZE=50
	predictions = model.predict(test_images, batch_size=BATCH_SIZE)
	
	feature_1_pred=np.argmax(predictions[0],axis=1)
	feature_2_pred=np.argmax(predictions[1],axis=1)
	feature_3_pred=np.argmax(predictions[2],axis=1)
	feature_4_pred=np.argmax(predictions[3],axis=1)

	feature_1_true=np.array([np.where(test_labels[i,0:9])[0].astype(int)[0] for i  in range(test_labels.shape[0])])
	feature_2_true=np.array([np.where(test_labels[i,9:18])[0].astype(int)[0] for i  in range(test_labels.shape[0])])
	feature_3_true=np.array([np.where(test_labels[i,18:39])[0].astype(int)[0] for i  in range(test_labels.shape[0])])
	feature_4_true=np.array([np.where(test_labels[i,39:50])[0].astype(int)[0] for i  in range(test_labels.shape[0])])

	cnf_matrix1=confusion_matrix(feature_1_true,feature_1_pred)
	cnf_matrix2=confusion_matrix(feature_2_true,feature_2_pred)
	cnf_matrix3=confusion_matrix(feature_3_true,feature_3_pred)
	cnf_matrix4=confusion_matrix(feature_4_true,feature_4_pred)
	####################
	####################
	####################
	normalize=True
	####################
	####################
	####################
	plt.figure()
	plot_confusion_matrix(cnf_matrix1,classes=[str(i) for i in range(9)],
							title='Confusion Matrix Feature 1',normalize=normalize)
	plt.savefig(visdir+'/CFM1.png',dpi=140,transparent=True)

	plt.figure()
	plot_confusion_matrix(cnf_matrix2,classes=[str(i) for i in range(9)],
							title='Confusion Matrix Feature 2',normalize=normalize)
	plt.savefig(visdir+'/CFM2.png',dpi=140,transparent=True)
	
	plt.figure()
	plot_confusion_matrix(cnf_matrix3,classes=[str(i) for i in range(21)],
							title='Confusion Matrix Feature 3',normalize=normalize)
	plt.savefig(visdir+'/CFM3.png',dpi=140,transparent=True)
	
	plt.figure()
	plot_confusion_matrix(cnf_matrix4,classes=[str(i) for i in range(11)],
							title='Confusion Matrix Feature 4',normalize=normalize)
	plt.savefig(visdir+'/CFM4.png',dpi=140,transparent=True)
	
	plt.show()