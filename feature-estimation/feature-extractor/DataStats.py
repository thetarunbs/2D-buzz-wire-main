import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os

def extractor(directory,labels_file):
	labels_df=pd.read_csv(directory+labels_file)
	filenames=labels_df.columns.values[1:]
	examples=np.asarray([cv2.imread(directory+file) for file in filenames])
	labels=np.asarray([(labels_df[file].values).astype(int) for file in filenames])
	return examples,labels

def class_counter(labels,feature=1):
	if feature==1:
		label_size=9
		start=0
	elif feature==2:
		label_size=9
		start=9
	elif feature==3:
		label_size=21
		start=18
	elif feature==4:
		label_size=11
		start=39
	else:
		print('Invalid Feature Choice')
		exit()

	classes=[i for i in range(label_size)]
	count=[0 for i in range(label_size)]
	labels=labels[:,start:start+label_size]
	indices=list(np.where(labels)[1])
	for i in indices:
		count[i]+=1
	class_count=dict(zip(classes,count))
	print('Class Count for Feature '+str(feature)+': ',class_count)
	histogrammer(class_count,feature)
	return class_count

def histogrammer(frequencies,feature):
	classes=list(frequencies.keys())
	freqs=frequencies.values()
	ax=plt.axes()
	ax.set_xticks(np.array(classes))
	plt.bar(classes,freqs,width=1.0,color='b')
	plt.xlabel('Classes')
	plt.ylabel('Frequency')
	plt.title('Frequency of classes for feature '+str(feature))
	#plt.show()


if __name__=='__main__':

	####################
	####################
	datadir=input('Dataset version directory: ')
	directory='Datasets/'+datadir
	####################
	####################
	training_directory=directory+'/training/'
	testing_directory=directory+'/testing/'
	training_labels='1. train_labels.csv'
	testing_labels='1. test_labels.csv'

	train_images,train_labels=extractor(training_directory,training_labels)
	print('Training Data Loaded')
	test_images,test_labels=extractor(testing_directory,testing_labels)
	print('Testing Data Loaded')
	
	statsdir=directory+'/DataStats/trainstats/'
	try:
		os.makedirs(statsdir)
		print('Directory created: ',statsdir)
	except FileExistsError:
		print('Directory ',statsdir,' Already exists')
	
	for feature in range(1,5):
		class_counter(train_labels,feature)
		plt.savefig(statsdir+'feature_'+str(feature)+'.png',dpi=140,transparent=True)
		plt.close()
	print('Training Statistics made')

	statsdir=directory+'/DataStats/teststats/'
	try:
		os.makedirs(statsdir)
		print('Directory created: ',statsdir)
	except FileExistsError:
		print('Directory ',statsdir,' Already exists')
	
	for feature in range(1,5):
		class_counter(test_labels,feature)
		plt.savefig(statsdir+'feature_'+str(feature)+'.png',dpi=140,transparent=True)
		plt.close()
	print('Testing Statistics made')