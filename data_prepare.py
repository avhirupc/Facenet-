import shutil,glob,os
from glob import glob

def makemydir(path):
  try:
    os.makedirs(path)
  except OSError:
    pass
  
def prepare_data():
	#makemydir('data')
	#os.makedirs('data')
	list_of_folders=glob('dataface /*')
	imp_folder=[]
	
	for folder in list_of_folders:
		subd = [s.rstrip("/") for s in glob(folder+"/*") if os.path.isdir(s)]
		imp_folder.append(subd)

	print len(imp_folder[0])
	#for train
	for subd in imp_folder[0]:
		try:
			shutil.copytree(subd+'/face','data/'+subd[9:])
		except:
			continue

	#for valid
	for subd in imp_folder[1]:
		try:
			shutil.copytree(subd+'/face','data/'+subd[9:])
		except:
			continue

if __name__=='__main__':
	prepare_data()