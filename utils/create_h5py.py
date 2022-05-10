from PIL import Image 
import numpy as np
import h5py as hf
import os

def Readimage(path):
    return np.array(Image.open(path).convert('RGB').resize((128,128)))


dataset_name = './checkpoints/imagenetmini_val'
dataset_path=r'/data/public/imagenet-mini/val'


f = hf.File( dataset_name+'.h5', 'w')

sub_dir = os.listdir(dataset_path)

for i,dir in enumerate(sub_dir):
    sub_data_length = len(os.listdir(os.path.join(dataset_path,dir)))
    g = f.create_group(dir)
    d = g.create_dataset(name=dir,shape=(sub_data_length,128,128,3),dtype=np.uint8)
    for i,_ in enumerate(d):
        name = os.listdir(os.path.join(dataset_path,dir))[i]
        d[i,:,:,:] = Readimage(os.path.join(dataset_path,dir,name))
        print(dir,name)
f.close()


f = hf.File(dataset_name+'.h5', 'r')
print(f.keys())
f.close()