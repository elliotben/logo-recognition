from PIL import Image

from resizeimage import resizeimage
import os
from os import listdir
from os.path import isfile, join
import subprocess

companies = ['McDo', 'Starbucks', '5_guys', 'ABP', 'Dunkin']

RESIZED_HEIGHT = 90
RESIZED_WIDTH = 90

for comp in companies:
    mypath = './' + comp
    #get list of images
    images = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    #resize and save new image
    for image in images:
        cur_dir = mypath + '/'
        image_path =  cur_dir + image
        if image_path == cur_dir + '.DS_Store' or image_path.endswith('lst'):
            continue
        im1 = Image.open(image_path)
        im1 = im1.convert('RGB')
        #use nearest neighbour to resize
        im2 = im1.resize((RESIZED_WIDTH, RESIZED_HEIGHT), Image.NEAREST)
        ext = ".jpg"
        im2.save(image_path)


    for image in images:
        cur_dir = mypath + '/'
        image_path =  cur_dir + image
        if image_path == cur_dir + '.DS_Store':
            continue
        #creates samples
        os.system('opencv_createsamples -img ' + image_path + ' -bg new_bg.txt\
         -info ' + comp + '/info.lst -maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.5 -num 1500')
        #os.rename(image_path, "../original images/" + image)
        #change names of created samples (opencv doesnt allow to specify prefix for new samples)
        for f in os.listdir(mypath):
            if not f.startswith('.') and not f in images and not f.endswith('lst'):
                os.chdir(comp)
                os.rename(f, image[:-4]+str(f))
                os.chdir('../')
