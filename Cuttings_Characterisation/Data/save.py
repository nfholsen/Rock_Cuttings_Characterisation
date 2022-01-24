__author__ = 'nolsen'

import glob, getopt, sys
import numpy as np
import sys, os

from skimage.io import imsave
from skimage import img_as_ubyte

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from Lamp.AttrDict.AttrDict import *
from Lamp.preprocessing.preprocessing import *

def save_to_png(image,path_save,**kwargs):

    image_name = image.load_path.split('\\')[1].split('.')[0]
    
    if kwargs:
        if 'type' in kwargs:
            if 'name' in kwargs:                
                for i_, region in enumerate(image.regions):
                    imToSave = region[kwargs['type']][kwargs['name']]

                    imToSave = img_as_ubyte(imToSave)

                    fname = f"{path_save}/{image_name}_{kwargs['type']}_{kwargs['name']}_{str(i_+1)}.png"
                    imsave(fname,imToSave)

            else:
                for i_, region in enumerate(image.regions):
                    imToSave = region[kwargs['type']]['new']

                    imToSave = img_as_ubyte(imToSave)

                    fname = f"{path_save}/{image_name}_{kwargs['type']}_raw_{str(i_+1)}.png"
                    imsave(fname,imToSave)

"""
Description :

This script will extract all the cuttings from a given list of scans and save them in a folder. 
The input file with the .yaml format will contain all the parameters to avoid redundancy.

"""
def main():
    myopts, args = getopt.getopt(sys.argv[1:],"i:o:")

    ifile=''
    ofile='None'

    for o, a in myopts:
        if o == '-i':
            ifile=a
        elif o == '-o':
            ofile=a
        else:
            print("Usage: %s -i input -o output" % sys.argv[0])

    if os.path.isfile(ifile) and os.path.splitext(ifile)[-1] in [".yaml",".yml"]:
        inputs = AttrDict.from_yaml_path(ifile) # change to argv
    else:
        raise AssertionError("Wrong input type")

    ### Read Data ###
    path_load = os.path.dirname(inputs.rootLoad + inputs.folderLoad)

    ### Save Outputs ###
    path_save = os.path.dirname(inputs.rootSave + inputs.folderSave)

    files_types = inputs.imgTypes

    lists_of_files = [glob.glob(path_load + type_) for type_ in files_types]
    list_of_files = [item for elem in lists_of_files for item in elem]

    ### Check if folder already exists ###
    if not os.path.isdir(path_save):
            os.mkdir(path_save)

    inFile = inputs.inFile
    outFile = inputs.outFile
    step = inputs.step

    for file in list_of_files[inFile:outFile:step]: # TODO : Change with an abstract class from transform

        image_raw = Image_Raw(file,'unchanged')

        # First
        Custom().apply(image = image_raw, thres = Otsu().return_threshold(image_raw))

        # Second 
        selem = disk(2)
        Erosion(selem=selem).apply(image_raw)

        # Third
        Mask().apply(image_raw)
        Labelize().apply(image_raw,threshold=5000)

        # Fourth
        selem = disk(5)
        Closing(selem=selem).apply_regions(image_raw)

        # Fifth
        selem = disk(2)
        Erosion(selem=selem).apply(image_raw, 3)

        # Sixth
        Mask().apply(image_raw)
        Labelize().apply(image_raw,threshold=5000)

        # Seventh
        selem = disk(2)
        Dilation(selem=selem).apply(image_raw,4)

        # Eigth
        Rescale_Intenstiy().apply_regions(image_raw, in_range=(1,99), out_range = (0,1), background=False)

        define_regions(image_raw)

        #Resize(out_shape=256).apply_regions(image_raw,type='bbox',name='resized')

        #Resize(out_shape=256).apply_regions(image_raw,type='mar',name='resized')

        #Padding(out_shape=256).apply_regions(image_raw,type='bbox',name='padded')

        #Padding(out_shape=256).apply_regions(image_raw,type='mar',name='padded')

        # Save BBOX
        save_to_png(image_raw, path_save=path_save, type='bbox')
        #save_to_png(image_raw, path_save=path_save, type='bbox',name='padded')
        #save_to_png(image_raw, path_save=path_save, type='bbox',name='resized')

        # Save MAR
        save_to_png(image_raw, path_save=path_save, type='mar')
        #save_to_png(image_raw, path_save=path_save, type='mar',name='padded')
        #save_to_png(image_raw, path_save=path_save, type='mar',name='resized')

if __name__ == "__main__":
    main()