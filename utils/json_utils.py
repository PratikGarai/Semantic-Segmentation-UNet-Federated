import json
from PIL import ImageColor


def get_classes(path : str) :
    f = open(path+"/classes.json", "r")
    data = json.load(f)["classes"]

    class_names = []
    color_dict = {}
    for i in data : 
        class_names.append(i["title"])
        r, g, b = ImageColor.getcolor(i["color"], "RGB")
        color_dict[i["title"]] = (b, g, r)

    return color_dict, class_names



def main() :
    class_names, color_dict =  get_classes("../data")
    print(class_names)
    print(color_dict)

if __name__=="__main__" :
    main()