import cv2 
import numpy as np
from glob import glob
from utils.json_utils import get_classes
# from json_utils import get_classes


def get_image_names(root : str) :
    return sorted(glob(root + '/images/*.png'))


def get_image_and_mask(img_path : str, color_dict : dict, classes : list) :
    mask_path = img_path.replace('images', 'masks')
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path)

    cls_mask = np.zeros(mask.shape)
    for class_name, bgr in color_dict.items() : 
        cls_mask[mask == bgr] = classes.index(class_name) + 1
    cls_mask = cls_mask[:,:,0] 

    return image, cls_mask



def main() :
    image_path = get_image_names("../data")[0]
    mask_path = image_path.replace("images", "masks")
    mask = cv2.imread(mask_path)
    color_dict, class_names = get_classes("../data")
    _, mask = get_image_and_mask(image_path, color_dict, class_names)
    with open("tmp.txt", "w+") as f : 
        for i in mask :
            for j in i : 
                f.write(f"{str(j)} ")
            f.write("\n")

    cv2.imshow("Mask : ", mask)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=="__main__" :
    main()