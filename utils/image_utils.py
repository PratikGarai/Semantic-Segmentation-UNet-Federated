import cv2 
import numpy as np

def get_image_and_masks(img_path : str, color_dict : dict, classes : list) :

    mask_path = img_path.replace('images', 'masks')
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    cls_mask = np.zeros(mask.shape)
    for class_name, bgr in color_dict.items() : 
        cls_mask[mask == bgr] = classes.index(class_name)
    cls_mask = cls_mask[:,:,0] 

    return image, cls_mask