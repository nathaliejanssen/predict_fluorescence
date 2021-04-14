import os
import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf

def create_image_path_df(path_to_imgfolder):
    img_path = Path(path_to_imgfolder)

    brightfield = [x.as_posix() for x in img_path.glob('*ch4*.png')]
    brightfield.sort()

    nucleus = [x.as_posix() for x in img_path.glob('*ch1*.png')]
    nucleus.sort()

    df_imgs = pd.DataFrame(data={'nucleus': nucleus, 'brightfield': brightfield})

    return df_imgs

def read_imgs(nucleus, brightfield, plane, number_of_slices, plane_string):
    # Define plane first image in the stack
    start = plane - int(number_of_slices/2)
    
    if start < 10:
        start_string = tf.strings.format('p0{}', start)
    else:
        start_string = tf.strings.format('p{}', start)

    # Prepare stack of bf images
    imgs = tf.TensorArray(tf.float32, size=number_of_slices)
    
    for i in range(number_of_slices):
        fn = tf.strings.regex_replace(brightfield, plane_string, start_string)

        if i > 0:
            if start < 10:
                next_plane = tf.strings.format('p0{}', start+i)
            else:
                next_plane = tf.strings.format('p{}', start+i)
            
            fn = tf.strings.regex_replace(fn, start_string, next_plane)

        img = tf.io.decode_png(tf.io.read_file(fn), channels=1, dtype=tf.uint16) 
        img = tf.image.convert_image_dtype(img, tf.float32)
        imgs = imgs.write(i, tf.squeeze(img))

    bfimg = imgs.stack()    
    bfimg = tf.transpose(bfimg, perm=[1,2,0])

    # Prepare fluorescent target
    fluo_target = tf.io.decode_png(tf.io.read_file(nucleus), channels=1,dtype=tf.uint16)
    fluo_target = tf.image.convert_image_dtype(fluo_target, tf.float32)

    return bfimg, fluo_target


df_train = create_image_path_df("C:/Users/natha/hello/spheroids/spheroids_png_crop")

df_train['plane'] = df_train['brightfield'].apply(lambda x: int(x.split('-')[0].split('p')[-1]))
df_train['plane_string'] = df_train['brightfield'].apply(lambda x: str(x.split('-')[0].split('f01')[-1]))
df_train = df_train[(df_train.plane<29) & (df_train.plane>2)]   # TODO: this is dependent on the stack that is provided. Now this is specific for this data set. 

df_train['number_of_slices'] = 5

df_train = df_train.astype({col: 'int32' for col in df_train.select_dtypes('int64').columns})       # convert to int32

ds_train = tf.data.Dataset.from_tensor_slices((df_train['nucleus'],
                                                df_train['brightfield'], 
                                                df_train['plane'],
                                                df_train['number_of_slices'],
                                                df_train['plane_string']))

ds_train = ds_train.map(read_imgs)

# print(list(ds_train.as_numpy_iterator()))
for stack, target in ds_train.take(1):
  print (stack.numpy())
  print(target.numpy())