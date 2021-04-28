import os
import numpy as np
from pathlib import Path
import pandas as pd
import tensorflow as tf


def create_image_path_df(path_to_imgfolder):
    img_path = Path(path_to_imgfolder)
    brightfield = [x.as_posix() for x in img_path.glob('*Z01C04.png')]
    brightfield.sort()
    nucleus = [x.as_posix() for x in img_path.glob('*Z01C01.png')]
    nucleus.sort()
    df_imgs = pd.DataFrame(data={'nucleus': nucleus, 'brightfield': brightfield})

    return df_imgs

def read_imgs(nucleus, brightfield, plane, number_of_slices): 
    imgs = tf.TensorArray(tf.float32, size = number_of_slices)

    for i in range(number_of_slices):
        fn = brightfield
        if i > 0:
            fn = tf.strings.regex_replace(fn,'Z01', tf.strings.format('Z0{}', plane + i))
        img = tf.io.decode_png(tf.io.read_file(fn), channels=1, dtype=tf.uint16)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.crop_to_bounding_box(img, 0, 0, 512, 512)
        imgs = imgs.write(i, tf.squeeze(img))
    bfimg = imgs.stack()
    bfimg = tf.transpose(bfimg, perm=[1,2,0])

    fluo_target = tf.io.decode_png(tf.io.read_file(nucleus), channels = 1,dtype = tf.uint16)
    fluo_target = tf.image.convert_image_dtype(fluo_target, tf.float32)
    fluo_target = tf.image.crop_to_bounding_box(fluo_target, 0, 0, 512, 512)

    return bfimg, fluo_target


def main():
    bs = 1
    df_train = create_image_path_df("testdata_png/")
    df_train = df_train.astype({col: 'str' for col in df_train.select_dtypes('float64').columns})
    df_train['plane'] = 1
    df_train['number_of_slices'] = 7
    df_train = df_train.astype({col: 'int32' for col in df_train.select_dtypes('int64').columns})

    ds_train = tf.data.Dataset.from_tensor_slices((df_train['nucleus'],
                                                    df_train['brightfield'], 
                                                    df_train['plane'],
                                                    df_train['number_of_slices']))

    ds_train = ds_train.map(read_imgs)
    ds_train = ds_train.batch(batch_size = bs, drop_remainder = True)
    ds_train = ds_train.repeat()
    ds_train = ds_train.prefetch(buffer_size = tf.data.AUTOTUNE)


    model = tf.keras.models.load_model("unet_test.h5")
    sgd_learning_rate = tf.keras.optimizers.SGD(learning_rate=10e-3)
    model.compile(loss = 'mean_squared_error', optimizer = sgd_learning_rate, metrics = ['accuracy'])
    model.summary()

    my_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='models/c01/model.{epoch:02d}-{val_loss:.2f}.h5'),
        tf.keras.callbacks.TensorBoard(log_dir='logs/logs_c01')
    ]

    model.fit(ds_train,
                epochs = 15, 
                steps_per_epoch = df_train.shape[0] // bs, 
                validation_data = ds_train,
                validation_steps = df_train.shape[0] // bs,
                callbacks = my_callbacks, 
                shuffle = True)
    
    model.save("trained_unet_testdata_c01.h5")


if __name__=="__main__":
    main()
