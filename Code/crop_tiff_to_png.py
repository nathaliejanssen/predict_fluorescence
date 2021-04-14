import skimage.io
import pandas as pd
import os

def read_and_format_input(path_to_input):
    df = pd.read_csv(path_to_input, delimiter = " ", header = None, names = ["spheroidID", "index", "x", "y"])
    long_coordinates = df[::2]
    wide_coordinates = long_coordinates.pivot(index='spheroidID', columns='index').reset_index()
    
    return wide_coordinates

def get_centroid_coordinates(input):
    """ Calculate the pixel values of the centroid of each binary spheroid 
    """
    cols = ['spheroid', 'x1', 'x2', 'y1', 'y2']
    lst = []
    
    for i in range(0, len(input)):
        lst.append([input.iloc[i,0].rsplit('f0',1)[0], input.iloc[i,1], input.iloc[i,2], input.iloc[i,3], input.iloc[i,4]])
    
    df = pd.DataFrame(lst, columns = cols)

    df['width'] = df['x2'] - df['x1']
    df['height'] = df['y2'] - df['y1']

    df['x_center'] = round(df['x1'] + (df['width'] / 2))
    df['y_center'] = round(df['y1'] + (df['height'] / 2))

    return df

def calculate_crop_size(centroid_table, margin):  
    """ Calculate the max size to use for cropping
    """ 
    x = max(centroid_table['width']) + margin
    y = max(centroid_table['height']) + margin
    
    cropsize = max(x,y)
    return cropsize


raw_coordinates = read_and_format_input('Docs/pilot_coordinates.txt')
boundingbox_coord = get_centroid_coordinates(raw_coordinates)
cropsize = calculate_crop_size(boundingbox_coord, margin = 100)

image_directory = 'Data/Raw/spheroids/'
target_directory = 'Data/Processed/spheroids_png_crop/'
count = 1

for filename in os.listdir(image_directory):
    img_name = filename.split(".")[0]
    for spheroid in list(boundingbox_coord.spheroid):    # spheroid gets corresponding coords
        coordinates = boundingbox_coord[boundingbox_coord['spheroid'].str.contains(spheroid)]
        if filename.rsplit('f0',1)[0].startswith(spheroid):
            x1, x2 = int(coordinates.loc[:, 'x_center'] - (cropsize/2)), int(coordinates.loc[:, 'x_center'] + (cropsize/2))
            y1, y2 = int(coordinates.loc[:, 'y_center'] - (cropsize/2)), int(coordinates.loc[:, 'y_center'] + (cropsize/2))
            
            img = skimage.io.imread(image_directory + filename)
            crop_img = img[y1:y2, x1:x2]
            print(crop_img)

            skimage.io.imsave(target_directory+img_name+'.png', crop_img)
            
            print('Images cropped:', count,'/300')
            count += 1