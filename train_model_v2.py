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

def read_imgs(nucleus, brightfield, plane, number_of_slices):
    start_plane = plane - int(number_of_slices//2)
    imgs = tf.TensorArray(tf.float32, size = number_of_slices)

    for i in range(number_of_slices):
        if i == 0:
            if start_plane < 10:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p0{}', start_plane))
            else:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p{}', start_plane))
        else:
            next_plane = start_plane + i
            if next_plane < 10:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p0{}', next_plane))
            else:
                fn = tf.strings.regex_replace(brightfield, tf.strings.format('p0{}', plane) if plane < 10 else tf.strings.format('p{}', plane), tf.strings.format('p{}', next_plane))

        img = tf.io.decode_png(tf.io.read_file(fn), channels = 1, dtype = tf.uint16) 
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize_with_crop_or_pad(img, 848, 848)   # otherwise it won't work in the current unet architecture
        imgs = imgs.write(i, tf.squeeze(img))
    bfimg = imgs.stack()    
    bfimg = tf.transpose(bfimg, perm = [1,2,0])

    fluo_target = tf.io.decode_png(tf.io.read_file(nucleus), channels = 1,dtype = tf.uint16)
    fluo_target = tf.image.convert_image_dtype(fluo_target, tf.float32)
    fluo_target = tf.image.resize_with_crop_or_pad(fluo_target, 848, 848)   # otherwise it won't work in the current unet architecture

    return bfimg, fluo_target

def main():
    number_of_slices = 5
    bs = 1

    df_train = create_image_path_df("Data/Processed/spheroids_png_crop")
    df_train = df_train.astype({col: 'str' for col in df_train.select_dtypes('float64').columns})
    df_train['plane'] = df_train['brightfield'].apply(lambda x: int(x.split('-')[0].split('p')[-1]))
    df_train = df_train[(df_train.plane <= (df_train.plane.max() - (number_of_slices//2))) & (df_train.plane >= (df_train.plane.min() + (number_of_slices//2)))]
    df_train['number_of_slices'] = number_of_slices
    df_train = df_train.astype({col: 'int32' for col in df_train.select_dtypes('int64').columns})

    training_df = df_train[0:26]    # NOTE: this is ONLY for the pilot
    testing_df = df_train[26:]

    ds_train = tf.data.Dataset.from_tensor_slices((training_df['nucleus'],
                                                    training_df['brightfield'], 
                                                    training_df['plane'],
                                                    training_df['number_of_slices']))
    ds_train = ds_train.map(read_imgs)
    ds_train = ds_train.batch(batch_size = bs, drop_remainder = True)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((testing_df['nucleus'],
                                                    testing_df['brightfield'], 
                                                    testing_df['plane'],
                                                    testing_df['number_of_slices']))
    ds_test = ds_test.map(read_imgs)
    ds_test = ds_test.batch(batch_size = bs, drop_remainder = True)
    ds_test = ds_test.prefetch(buffer_size = tf.data.AUTOTUNE) 

    model = tf.keras.models.load_model("unet.h5")
    sgd_learning_rate = tf.keras.optimizers.SGD(learning_rate=10e-4)
    model.compile(loss = 'mean_squared_error', optimizer = sgd_learning_rate, metrics = ['accuracy'])
    model.summary()

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='models/model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')
    ]

    model.fit(ds_train,
                epochs = 3, 
                steps_per_epoch = training_df.shape[0] // bs, 
                validation_data = ds_test,
                validation_steps = testing_df.shape[0] // bs,
                callbacks = my_callbacks, 
                shuffle = True)

if __name__=="__main__":
    main()
