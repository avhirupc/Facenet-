
import shutil
from glob import glob
import random
from PIL import Image
import numpy

class TripletPair(object):
	"""docstring for TripletPair"""
	def __init__(self, anchor,positive,negative):
		self.anchor = anchor
		self.positive = positive
		self.negative = negative
	def __str__(self):
		return self.anchor+':'+self.positive+':'+self.negative

def get_label(data_dir):
	"""Given a data dir ,finds all label(name of identities)
    """
	all_label_path=glob(data_dir+'/*')
	labels={}
	l=len(data_dir)+1
	for i,label_path in enumerate(all_label_path):
		labels[label_path[l:]]=i
	return labels

def get_all_images_path(data_dir):
	"""Given a data dir ,returns all image paths
    Args: 
      data_dir
    Returns:
      dictionary of celebity name and respective iamges 
    """
	
	l=len(data_dir)+1
	
	all_images_path=glob(data_dir+'/*/*.jpg')
	labels=get_label(data_dir)
	labels_and_images={}
	labels=get_label(data_dir)
	for image_path in glob(data_dir+'/*'):
		labels_and_images[labels[image_path[l:]]]=glob(image_path+'/*.jpg')
	return labels_and_images

def get_minibatches(data_dir,mini_batch_size=90,minimum_per_identity=30):
	"""Given a data dir ,forms batches of image paths to be used later
    Args: 
      mini_batch_size : batch size
      minimum_per_identity : minimum number of personality per batch
    
    Constraint: 
    mini_batch_size%_minimum_per_identity= 0
    mini_batch_size%3 = 0       
    Returns:
      Numpy array of batches of image paths
    """
	if mini_batch_size%minimum_per_identity!=0:
		print("Choose approriate mini_batch_size and minimum_per_identity")
		return
	labels_and_images=get_all_images_path(data_dir)
	total=0
	for array in labels_and_images.values():
		total+=len(array)
	#print "Number of batches:",total/mini_batch_size
	batches=[]
	for i in range(total/mini_batch_size):	
		mini_batch=[]
		identity_added_batch=[]
		
		for key,values in labels_and_images.iteritems():
			if len(mini_batch)<=(mini_batch_size- minimum_per_identity):
				identity_added_batch.append(key)
				for value in values[0:minimum_per_identity]:
					mini_batch.append(value)
				labels_and_images[key]=values[minimum_per_identity:]
		batches.append(mini_batch)		
	return batches	

def get_batch_to_numpy(batch,mini_batch_size=90):
	"""Converts a batch of image paths into an batch of images
    Args:
      batch : batch of image paths
      mini_batch_size : batch size
           
    Returns:
      Return numpy array of images and their respective image paths
    """
	i=0
	temp_file_list=[]
	labels=[]
	for image in batch:
		#print triplet_pair
		temp_file_list.append(image)
		#print temp_file_list[0:2]
	batch_images = numpy.array([numpy.array(Image.open(fname).resize((220,220),Image.ANTIALIAS)) for fname in temp_file_list])
	batch_images= numpy.stack(batch_images,axis=0)
	return batch_images,batch


if __name__ == '__main__':
	triplet_batches=get_minibatches('/home/avhirup/Programming/Projects/Facenet/data/train')
	
	_,image_path=get_batch_to_numpy(triplet_batches[0])
	print image_path