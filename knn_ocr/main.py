import numpy as np
import cv2 
from skimage.measure import regionprops, label
from skimage.io import imread
from pathlib import Path

path = Path("task")

def extractor(image):
    if image.ndim == 2:
        gray = image
        binary = gray > 0
    else:
        gray = np.mean(image, axis=2)
        binary = gray > 0
    labeled = label(binary)
    regions = regionprops(labeled)
    
    
    if len(regions) > 1:
        
        main_region = max(regions, key=lambda r: r.area)
        
        
        features = [main_region.eccentricity * 0.95,
                     main_region.solidity, main_region.extent, 
                     (main_region.perimeter/main_region.area)*1.1, 
                     main_region.area_convex/main_region.area]
    else:
        
        features = [regions[0].eccentricity,
                    regions[0].solidity, 
                    regions[0].extent, 
                    regions[0].perimeter/regions[0].area,
                    regions[0].area_convex/regions[0].area]  
                                                                                                                                 
    return np.array(features, dtype="f4")
                                                                                                                                 
    

def make_train(path):
    train = []
    target = []
    class_map = {} 
    index = -1

    for i in sorted(path.glob("**")):
        index += 1
        class_map[index] = str(i)[-1]  

        for j in sorted(i.glob("*.png")):
            train.append(extractor(imread(j)))
            target.append(index)

    train = np.array(train, dtype = "f4").reshape(-1, 5)
    target = np.array(target, dtype = 'f4').reshape(-1, 1)

    return train, target, class_map  



for i in range(7):
    image = imread(path / f"{i}.png")

    train, target, class_map = make_train(path/"train")  
    knn = cv2.ml.KNearest.create()
    knn.train(train, cv2.ml.ROW_SAMPLE, target)


    gray = np.mean(image, axis=2)
    binary = gray > 0
    labeled = label(binary.T)
    regions = regionprops(labeled)

    find = []

    for j, region in enumerate(regions):
        if regions[j].extent < 0.7:
            find.append(extractor(regions[j].image))
    find = np.array(find, dtype = "f4").reshape(-1,5)

    ret, result, neighbours, dist = knn.findNearest(find,  3)

    result_string = ""
    for res in result:
        result_string+= class_map[int(res.item())]

    print(f"img{i}: {result_string}")
    
