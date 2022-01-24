import numpy as np
import cv2
import matplotlib.pyplot as plt 
import matplotlib.patches as patches

from skimage.measure import regionprops, label, find_contours
from skimage.filters import threshold_otsu
from skimage.morphology import disk, erosion, binary_erosion, dilation, binary_dilation, closing, binary_closing, opening, binary_opening
from skimage import exposure
from skimage import img_as_float, img_as_ubyte


import warnings


from abc import abstractmethod

class Image_Raw:
    def __init__(self,load_path,flag='color'):
        """
        Inputs : 
            load_path : Abolute path to load an image
            flag : (grayscale : cv2.IMREAD_GRAYSCALE), (color : cv2.IMREAD_COLOR), (unchanged : cv2.IMREAD_UNCHANGED)

        """
        self.load_path = load_path
        self.flag = flag

        if flag == 'grayscale':
            # cv2.IMREAD_GRAYSCALE : Read Image as Grey Scale
            img = cv2.imread(self.load_path, cv2.IMREAD_GRAYSCALE)
        elif flag == 'unchanged':
            # cv2.IMREAD_UNCHANGED : Useful when read Image with Transparency Channel
            img = cv2.imread(self.load_path, cv2.IMREAD_UNCHANGED)
        elif flag == 'color':
            # cv2.IMREAD_COLOR : Transparency channel ignoreg even if present
            img = cv2.imread(self.load_path, cv2.IMREAD_COLOR)
        else :
            raise Exception(f"Wrong flag, expected [color, grayscale or unchanged] and got {self.flag}")

        self.img = cv2.normalize(img, None, alpha=0., beta=1., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.new = self.img # TODO : Add dictionnary of operations

    def update(self,array):

        self.new = array

    def mask_times_raw(self):
        """
        Update the "new" image by multiplying the last assigned mask with the original image
        """

        try :
            self.mask
        except AttributeError:
            raise AssertionError("No mask attribute found, use Mask() to create mask")

        self.new = self.mask * self.img

    def plot(self,save_path=None):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.img, cmap='gray')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path,bbox_inches='tight')
        plt.show()

    def plot_new(self,save_path=None):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.new, cmap='gray')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path,bbox_inches='tight')
        plt.show()

    def plot_labels(self,save_path=None):

        try :
            self.labels
        except AttributeError:
            raise AssertionError("No labels attribute found, use Labelise() to create labels")

        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(self.labels, cmap='gray')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path,bbox_inches='tight')
        plt.show()

    def plot_histogram(self):
        histogram, bin_edges = np.histogram(self.img, bins=256)
        plt.plot(bin_edges[0:-1], histogram)
        plt.title("Graylevel histogram")
        plt.xlabel("gray value")
        plt.ylabel("pixel count")
        plt.show()

    def plot_histogram_new(self):
        histogram, bin_edges = np.histogram(self.new, bins=256)
        plt.plot(bin_edges[0:-1], histogram)
        plt.title("Graylevel histogram")
        plt.xlabel("gray value")
        plt.ylabel("pixel count")
        plt.show()
    
    def plot_cfd(self):
        histogram, bin_edges = np.histogram(self.img, bins=256)
        return histogram, bin_edges

    def plot_cfd_new(self):
        histogram, bin_edges = np.histogram(self.new, bins=256)
        return histogram, bin_edges

def AssertImage(image):
    try : 
        assert isinstance(image, Image_Raw)
    except (AssertionError):
        raise AssertionError(f'Input variable should be of class Image but got : {type(image)}')

class Thresholding:
    def __init__():
        pass

    @abstractmethod
    def compute_threshold(self):
        pass

    def apply(self,image, **kwargs):

        AssertImage(image)

        thres, mask = self.compute_threshold(image, **kwargs)

        image.new = mask # Update image
        
    def return_mask(self,image, **kwargs):

        _, mask = self.compute_threshold(image, **kwargs) # Store computed mask in the object

        return mask

    def return_threshold(self,image, **kwargs):

        thres , _ = self.compute_threshold(image, **kwargs) # Store computed threshold in the object

        return thres

class Otsu(Thresholding):
    def __init__(self):
        pass
    
    def compute_threshold(self,image, **kwargs):

        thres, mask = cv2.threshold(np.uint8(image.new*255),0, 255, cv2.THRESH_OTSU)

        thres = thres/255

        return thres, mask

class Binary(Thresholding):
    """
    If pixel intensity is greater than the set threshold, value set to 255, else set to 0 (black).
    """
    def __init__(self):
        pass
    
    def compute_threshold(self,image,**kwargs):

        thres, mask = cv2.threshold(np.uint8(image.new*255), kwargs['thres']*255, 255, cv2.THRESH_BINARY)

        thres = thres/255

        return thres, mask

class Binary_Inverted(Thresholding):
    """
    Inverted or Opposite case of cv2.THRESH_BINARY.
    """
    def __init__(self):
        pass    
    
    def compute_threshold(self,image,**kwargs):

        thres, mask = cv2.threshold(np.uint8(image.new*255), kwargs['thres']*255, 255, cv2.THRESH_BINARY_INV)

        thres = thres/255

        return thres, mask

class Truncated(Thresholding):
    """
    If pixel intensity value is greater than threshold, it is truncated to the threshold. The pixel values are set to be the same as the threshold. All other values remain the same.
    """
    def __init__(self):
        pass

    def compute_threshold(self,image,**kwargs):

        thres, mask = cv2.threshold(np.uint8(image.new*255), kwargs['thres']*255, 255, cv2.THRESH_TRUNC)
        
        thres = thres/255

        return thres, mask

class To_Zero(Thresholding):
    """
    Pixel intensity is set to 0, for all the pixels intensity, less than the threshold value.
    """
    def __init__(self):
        pass

    def compute_threshold(self,image,**kwargs):

        thres, mask = cv2.threshold(np.uint8(image.new*255), kwargs['thres']*255, 255, cv2.THRESH_TOZERO)

        thres = thres/255

        return thres, mask

class To_Zero_Inverted(Thresholding):
    """
    Inverted or Opposite case of cv2.THRESH_TOZERO.
    """
    def __init__(self):
        pass

    def compute_threshold(self,image,**kwargs):

        thres, mask = cv2.threshold(np.uint8(image.new*255), kwargs['thres']*255, 255, cv2.THRESH_TOZERO_INV)

        thres = thres/255

        return thres, mask

class Custom(Thresholding):
    def __init__(self):
        pass

    def compute_threshold(self,image,**kwargs):
        """
        thres = Otsu().return_mask(image)
        """

        thres = kwargs['thres']

        cdf, _ = image.plot_cfd()
        cdf = np.cumsum(cdf)

        cdf = cdf/cdf[-1]

        xs = np.linspace(0,1,len(cdf))

        cdf_norm = (cdf[xs > thres] - cdf[xs > thres][0])  / (cdf[xs > thres]  - cdf[xs > thres][0])[-1]

        threshold_final = xs[xs > thres][cdf_norm > 0.05][0]

        mask = (image.new > threshold_final).astype(int)

        return threshold_final, mask

class Morphological:
    def __init__(self,selem):
        self.selem = selem
    
    @abstractmethod
    def apply(self):
        pass

class Erosion(Morphological):
    def __init__(self,selem):
        super().__init__(selem) 

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for i in range(args[0]):
                image.new = erosion(image.new, self.selem)
        else:
            image.new = erosion(image.new, self.selem)

class BinaryErosion(Morphological):
    def __init__(self,selem):
        super().__init__(selem)

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for i in range(args[0]):
                image.new = binary_erosion(image.new, self.selem)
        else:
            image.new = binary_erosion(image.new, self.selem)
        
class Dilation(Morphological):
    def __init__(self,selem):
        super().__init__(selem) 

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for _ in range(args[0]):
                image.new = dilation(image.new, self.selem)

                if hasattr(image,'mask'):
                    image.mask = dilation(image.mask, self.selem)
                if hasattr(image,'labels'):
                    image.labels = dilation(image.labels, self.selem) 
        else:
            image.new = dilation(image.new, self.selem)

            if hasattr(image,'mask'):
                image.mask = dilation(image.mask, self.selem)
            if hasattr(image,'labels'):
                image.labels = dilation(image.labels, self.selem) 

        # Update each region if there are :
        if hasattr(image,'regions'):
            for i in range(len(image.regions)):

                image.regions[i]['whole_mask'] = (image.labels == (i+1)).astype(int)
                image.regions[i]['whole_raw'] = image.regions[i]['whole_mask'] * image.img
                image.regions[i]['whole_new'] = image.regions[i]['whole_mask'] * image.img 

    def apply_regions(self,image,*args):

        AssertImage(image)

        for i_, region_i in enumerate(image.regions):

            if args:
                for _ in range(args[0]):
                    region_i['whole_mask'] = dilation(region_i['whole_mask'], self.selem)
            else:
                region_i['whole_mask'] = dilation(region_i['whole_mask'], self.selem)

            image.regions[i_]['whole_mask'] = region_i['whole_mask']

        # Update "labels" attribute
        image.labels = np.sum([region['whole_mask']*(i+1) for i, region in enumerate(image.regions)],axis=0)
        image.mask = np.sum([region['whole_mask'] for i, region in enumerate(image.regions)],axis=0)
        image.new = np.sum([region['whole_new'] for i, region in enumerate(image.regions)],axis=0)

class BinaryDilation(Morphological):
    def __init__(self,selem):
        super().__init__(selem)

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for i in range(args[0]):
                image.new = binary_dilation(image.new, self.selem)
        else:
            image.new = binary_dilation(image.new, self.selem)

class Closing(Morphological):
    def __init__(self,selem):
        super().__init__(selem)

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for _ in range(args[0]):
                image.new = closing(image.new, self.selem)

                if hasattr(image,'mask'):
                    image.mask = closing(image.mask, self.selem)
                if hasattr(image,'labels'):
                    image.labels = closing(image.labels, self.selem) 
        else:
            image.new = closing(image.new, self.selem)
            
            if hasattr(image,'mask'):
                image.mask = closing(image.mask, self.selem)
            if hasattr(image,'labels'):
                image.labels = closing(image.labels, self.selem)

        # Update each region if there are :
        if hasattr(image,'regions'):
            for i_ in range(len(image.regions)):
                image.regions[i_]['whole_mask'] = (image.labels == (i_+1)).astype(int)
                image.regions[i_]['whole_raw'] = image.regions[i_]['whole_mask'] * image.img
                image.regions[i_]['whole_new'] = image.regions[i_]['whole_mask'] * image.img 


    def apply_regions(self,image,*args):

        AssertImage(image)

        for i_, region_i in enumerate(image.regions):

            if args:
                for _ in range(args[0]):
                    region_i['whole_mask'] = closing(region_i['whole_mask'], self.selem)
            else:
                region_i['whole_mask'] = closing(region_i['whole_mask'], self.selem)

            image.regions[i_]['whole_mask'] = region_i['whole_mask']
            image.regions[i_]['whole_raw'] = image.regions[i_]['whole_mask'] * image.img
            image.regions[i_]['whole_new'] = image.regions[i_]['whole_mask'] * image.img 


        # Update "labels" attribute
        image.labels = np.sum([region['whole_mask']*(i+1) for i, region in enumerate(image.regions)],axis=0)
        image.mask = np.sum([region['whole_mask'] for i, region in enumerate(image.regions)],axis=0)
        image.new = image.mask

class BinaryClosing(Morphological):
    def __init__(self,selem):
        super().__init__(selem)

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for i in range(args[0]):
                image.new = binary_closing(image.new, self.selem)
        else:
            image.new = binary_closing(image.new, self.selem) 

class Opening(Morphological):
    def __init__(self,selem):
        super().__init__(selem)

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for i in range(args[0]):
                image.new = opening(image.new, self.selem)
        else:
            image.new = opening(image.new, self.selem) 

class BinaryOpening(Morphological):
    def __init__(self,selem):
        super().__init__(selem)

    def apply(self,image,*args):

        AssertImage(image)

        if args:
            for i in range(args[0]):
                image.new = binary_opening(image.new, self.selem)
        else:
            image.new = binary_opening(image.new, self.selem)            

class Exposure:
    def __init__(self):
        pass

    @abstractmethod
    def apply(self):
        pass

class Rescale_Intenstiy(Exposure):
    def __init__(self):
        pass

    def apply(self, image : Image_Raw, in_range : tuple = (1,99), out_range : tuple = (0,1), background : bool = False):
        """
        Apply rescale intensity on the whole current new image
        p1, p99 = np.percentile(region_i_raw[region_i_raw > 0], (1, 99))
        in range (Tuple) : percentage to keep
        """

        AssertImage(image)

        if background :
            p_down, p_up = np.percentile(image.new, in_range)

            image.new = exposure.rescale_intensity(image.new, in_range=(p_down, p_up))

        else: 
            p_down, p_up = np.percentile(image.new[image.new > 0], in_range) # omit 0 values for the background

            image.new = exposure.rescale_intensity(image.new, in_range=(p_down, p_up), out_range = out_range)

    def apply_regions(self, image : Image_Raw, in_range : tuple = (1,99), out_range : tuple = (0,1), background : bool = False):

        AssertImage(image)

        for i_, region in enumerate(image.regions):

            if background :
                p_down, p_up = np.percentile(region['whole_new'], in_range)
            else:
                p_down, p_up = np.percentile(region['whole_new'][region['whole_new'] > 0], in_range)

            image.regions[i_]['whole_new'] = exposure.rescale_intensity(region['whole_new'], in_range=(p_down, p_up))

            if np.min(image.regions[i_]['whole_new']) < 0 or np.max(image.regions[i_]['whole_new']) > 1:

                image.regions[i_]['whole_new'][image.regions[i_]['whole_new'] > 1] = 1
                image.regions[i_]['whole_new'][image.regions[i_]['whole_new'] < 0] = 0

            if np.min(image.regions[i_]['whole_raw']) < 0 or np.max(image.regions[i_]['whole_raw']) > 1:

                image.regions[i_]['whole_raw'][image.regions[i_]['whole_raw'] > 1] = 1
                image.regions[i_]['whole_raw'][image.regions[i_]['whole_raw'] < 0] = 0
        
        image.new = np.sum([region['whole_new'] for _, region in enumerate(image.regions)],axis=0)

class Mask:
    """

    Save a copy of the current image state that is considered to be the mask

    --> Each time a new morphological operation will be applied "mask" will be updated"

    """ 
    def __init__(self):
        pass

    def apply(self, image):
        
        AssertImage(image)

        # TODO : Assert Binary

        image.mask = image.new

class Labelize:
    """

    Update the current image with labels and also save a new image with the labels 
    
    --> Allow to work at the region level (Each operation at the "region" level will update the "mask","labels")

    """
    def __init__(self):
        pass

    def apply(self, image, **kwargs):
        
        AssertImage(image)

        # TODO : Add assertion that mask is binary

        image.labels = label(image.mask) # Labelize

        regions = regionprops(image.labels) 

        image.regions = []

        if kwargs:
            for i in range(len(regions)):

                region_i = regions[i]

                if region_i.area > kwargs['threshold'] :

                    region_whole_mask = image.labels == region_i.label # Find region on the whole image corresponding to the label of the region
                    region_whole_raw = region_whole_mask * image.img

                    image.regions.append(
                        {
                            'whole_mask' : region_whole_mask.astype(int),
                            'whole_raw' : region_whole_raw,
                            'whole_new' : region_whole_raw
                        })

            # Update labels :
            image.labels = np.sum([region['whole_mask']*(i+1) for i, region in enumerate(image.regions)],axis=0)
        
        else:

            warnings.warn(f"{len(regions)} regions were detected and no threshold was set, this can take some time")

            for i in range(len(regions)):

                region_i = regions[i]

                region_whole_mask = image.labels == region_i.label
                region_whole_raw = region_whole_mask * image.img

                image.regions.append(
                    {
                        'whole_mask' : region_whole_mask.astype(int),
                        'whole_raw' : region_whole_raw,
                        'whole_new' : region_whole_raw
                    })

def crop_rectangle(im_rect,box,rect):
    """
    For Minimum Area Rectangle
    """
    W = rect[1][0]
    H = rect[1][1]
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    
    rotated = False
    angle = rect[2]
    
    if angle < -45:
        angle+=90
        rotated = True

    if angle > 45:
        angle-=90
        rotated = True
        
    size = (x2-x1,y2-y1)
    center = (int((x1+x2)/2), int((y1+y2)/2))
    
    M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
    
    cropped = cv2.getRectSubPix(im_rect, size, center)    
    cropped = cv2.warpAffine(cropped, M, size)
    
    croppedW = W if not rotated else H 
    croppedH = H if not rotated else W
    
    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0]/2, size[1]/2))
    return croppedRotated

def define_regions(image,**kwargs):
    """

    Save both bounding box and minimum are for the moment for each region

    --> Once "define regions" has been called we can apply transform on each

    """

    ### Bounding Box ###

    try:
        image.labels
    except AttributeError:
        raise AssertionError(f'No label mask was assigned to the {type(image)} object, call Labelize')

    regions = regionprops(image.labels) # Faster to find all the regions from the label

    for i, region in enumerate(image.regions):
        
        minr, minc, maxr, maxc = regions[i].bbox

        box = np.array([[minc, minr],[minc, maxr],[maxc, maxr],[maxc,minr]])

        region_i_mask = region['whole_mask'][minr:maxr, minc:maxc] # Only mask of the region
        region_i_raw = region['whole_raw'][minr:maxr, minc:maxc] # Only raw of the region
        region_i_new = region['whole_new'][minr:maxr, minc:maxc]

        image.regions[i]['bbox'] = { 'box':box, 'mask':region_i_mask, 'raw':region_i_raw, 'new':region_i_new}

    ### Minimum Area Rectangle ###

    for i, region in enumerate(image.regions):

        mask = img_as_ubyte(region['whole_mask'])
        raw = img_as_ubyte(region['whole_raw'])
        new = img_as_ubyte(region['whole_new'])

        contours,_ = cv2.findContours(mask, 1, 2)

        rect = cv2.minAreaRect(contours[0])
        if len(contours) >= 2:
            cmax = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            rect = cv2.minAreaRect(cmax) 
        
        box = np.int0(np.round(cv2.boxPoints(rect)))

        region_i_mask = crop_rectangle(mask, box, rect)
        region_i_raw = crop_rectangle(raw, box, rect)
        region_i_new = crop_rectangle(new, box, rect)

        region_i_raw = region_i_raw/255
        region_i_new = region_i_new/255

        image.regions[i]['mar'] = {'box': box, 'mask':region_i_mask, 'raw':region_i_raw, 'new':region_i_new}



def plot_regions(image, **kwargs): # TODO : Change for "to save" plot meaning I have to add tranform at the sample level
    """
    Plot the regions in a image for the moment with only rectangular bbox
    """

    if 'type' in kwargs :

        for region in image.regions:

            region_i = region[kwargs['type']]

            fig,ax = plt.subplots(1,4,figsize=(20,5)) # Raw; To save; Distribution for To Save and Localisation

            box = region_i['box']

            region_i_raw = region_i['raw']

            if 'name' in kwargs:
                region_i_new = region_i[kwargs['name']]
            else:
                region_i_new = region_i['new']

            # Region low contrast
            ax[0].imshow(region_i_raw,
                cmap=plt.cm.gray,aspect="auto")
            ax[0].set_xlabel(f'Width : {region_i_raw.shape[1]} px')
            ax[0].set_ylabel(f'Height : {region_i_raw.shape[0]} px')
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # Region new (high contrast)
            ax[1].imshow(region_i_new,
                cmap=plt.cm.gray,aspect="auto")
            ax[1].set_xlabel(f'Width : {region_i_new.shape[1]} px')
            ax[1].set_ylabel(f'Height : {region_i_new.shape[0]} px')
            ax[1].set_xticks([])
            ax[1].set_yticks([])
                        
            # Region histogram + cfd
            ax[2].hist(region_i_new[region_i_new>0].ravel(), bins=256, histtype='step', color='black')
            ax[2].set_xlim(0, 1)
            ax[2].set_yticks([])

            ax_cdf = ax[2].twinx()

            img_cdf, bins = exposure.cumulative_distribution(region_i_new[region_i_new>0].ravel(), 256) # Remove the background
            ax_cdf.plot(bins, img_cdf, 'r')
            ax_cdf.set_yticks([])

            # Region on low contrast image
            poly = patches.Polygon(box,fill=False, edgecolor='white', linewidth=2)

            ax[3].imshow(image.img, cmap=plt.cm.gray,aspect="auto")
            ax[3].add_patch(poly)
            ax[3].axis(False)

            plt.show()

class Resize():
    """
    Resize image or regions, independently, work later to resize both accordingly 
    e.g. if we reisze the whole image then the regions will be resized too to keep the same aspect ratio
    """
    def __init__(self,out_shape : tuple or int):
        """
        Inputs : 
            - out_shape (tuple or int) : for tuple respectively width and height and for int dimension is broadcasted
        """
        
        if type(out_shape) is not tuple and type(out_shape) is not int :
            raise AssertionError(f'Expected type is wrong got {type(out_shape)} and tuple or int was expected')
        if type(out_shape) is int:
            self.width, self.height = out_shape, out_shape
        if type(out_shape) is tuple:
            self.width, self.height = out_shape[0], out_shape[1]

    def apply(self, image, **kwargs):
        """
        Resize image and update only the new with a cubic interpolation
        """

        if 'name' in kwargs: # Create "name" entry from "new"
            resized_image = cv2.resize(image.new, 
                                    (self.width, self.height), 
                                    interpolation=cv2.INTER_CUBIC)

            if np.min(resized_image) < 0 or np.max(resized_image) > 1:
                resized_image[resized_image > 1] = 1
                resized_image[resized_image < 0] = 0

            setattr(image,kwargs['name'],resized_image)

        else : # Update "new" 
            resized_image = cv2.resize(image.new, 
                                    (self.width, self.height), 
                                    interpolation=cv2.INTER_CUBIC)

            if np.min(resized_image) < 0 or np.max(resized_image) > 1:
                resized_image[resized_image > 1] = 1
                resized_image[resized_image < 0] = 0

            image.new = resized_image

    def apply_regions(self, image, **kwargs):
        """
        Resize region (for the moment bbox or mar, not the whole yet)
        """
        if 'type' in kwargs: # type is for bbox or mar
            for region in image.regions:

                if 'name' in kwargs: # Create "name" entry from "new"
                    resized_image = cv2.resize(region[kwargs['type']]['new'], 
                                    (self.width, self.height), 
                                    interpolation=cv2.INTER_CUBIC)

                    if np.min(resized_image) < 0 or np.max(resized_image) > 1:
                        resized_image[resized_image > 1] = 1
                        resized_image[resized_image < 0] = 0

                    region[kwargs['type']][kwargs['name']] = resized_image

                else: # Update "new" 
                    resized_image = cv2.resize(region[kwargs['type']]['new'], 
                                    (self.width, self.height), 
                                    interpolation=cv2.INTER_CUBIC) 

                    if np.min(resized_image) < 0 or np.max(resized_image) > 1:
                        resized_image[resized_image > 1] = 1
                        resized_image[resized_image < 0] = 0

                    region[kwargs['type']]['new'] = resized_image

        else :
            raise AssertionError('No type were detected')

class Padding():
    def __init__(self, out_shape : tuple or int):
        if type(out_shape) is not tuple and type(out_shape) is not int :
            raise AssertionError(f'Expected type is wrong got {type(out_shape)} and tuple or int was expected')
        if type(out_shape) is int:
            self.width, self.height = out_shape, out_shape
        if type(out_shape) is tuple:
            self.width, self.height = out_shape[0], out_shape[1]

    def pad(self, array, value):

        old_width, old_heigth = array.shape[:2]

        ratio = np.array([self.width,self.height]).astype(float)/np.array([old_width,old_heigth]).astype(float)

        new_size = tuple([int(x*min(ratio)) for x in [old_width,old_heigth]])

        resized_array = cv2.resize(array, (new_size[1], new_size[0]),interpolation=cv2.INTER_CUBIC)
    
        delta_w = self.width - new_size[1]
        delta_h = self.height - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        
        padded_array = cv2.copyMakeBorder(resized_array, top, bottom, left, right, cv2.BORDER_CONSTANT,value=int(value))
        return padded_array

    def apply(self, image, value : int = 0, **kwargs):
        """
        Inputs : 
            - image
            - value : background color or other (int)
        """
        if 'name' in kwargs: # Create "name" entry from "new" if available
            padded_image = self.pad(image.new, value=value)

            if np.min(padded_image) < 0 or np.max(padded_image) > 1:
                    padded_image[padded_image > 1] = 1
                    padded_image[padded_image < 0] = 0

            setattr(image,kwargs['name'],padded_image)

        else: # Update "new" 
            padded_image = self.pad(image.new, value=value)

            if np.min(padded_image) < 0 or np.max(padded_image) > 1:
                    padded_image[padded_image > 1] = 1
                    padded_image[padded_image < 0] = 0
        
            image.new = padded_image

    def apply_regions(self, image, value : int = 0, **kwargs):

        if 'type' in kwargs:
            for region in image.regions:

                if 'name' in kwargs: # Create "name" entry from "new" 
                    padded_image = self.pad(region[kwargs['type']]['new'], value=value)

                    if np.min(padded_image) < 0 or np.max(padded_image) > 1:
                        padded_image[padded_image > 1] = 1
                        padded_image[padded_image < 0] = 0

                    region[kwargs['type']][kwargs['name']] = padded_image

                else: # Update "new" 
                    padded_image = self.pad(region[kwargs['type']]['new'], value=value)

                    if np.min(padded_image) < 0 or np.max(padded_image) > 1:
                        padded_image[padded_image > 1] = 1
                        padded_image[padded_image < 0] = 0

                    region[kwargs['type']]['new'] = padded_image

        else :
            raise AssertionError('No type were detected')