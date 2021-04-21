from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, UpSampling2D, Concatenate

def down(connected_layer, num_filters, pool = True):
    conv = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(connected_layer)
    conv = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(conv)
    if pool == True:
        pool = MaxPool2D()(conv)
        return conv, pool
    else:
        return conv

def up(connected_layer, concat_layer, num_filters):
    up = UpSampling2D()(connected_layer)
    up = Conv2D(num_filters, kernel_size=(2, 2), padding="same")(up)
    up = Concatenate(axis=3)([concat_layer, up])
    conv = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(up)
    conv = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(conv)
    return conv

def create_unet_model():
    # "Down"
    unet_input = Input(shape = (848, 848, 5))
    c1, p1 =  down(unet_input, 64)
    c2, p2 = down(p1, 128)
    c3, p3 = down(p2, 256)
    c4, p4 = down(p3, 512)
    c5 = down(p4, 1024, pool = False)
    # "Up"
    c6 = up(c5, c4, 512)
    c7 = up(c6, c3, 256)
    c8 = up(c7, c2, 128)
    c9 = up(c8, c1, 64)
    out = Conv2D(filters=1, kernel_size=(1, 1), activation="relu")(c9)

    model = Model(inputs=unet_input, outputs = out, name = 'U-net')
    return model

model = create_unet_model()
# model.summary()
# save model to h5 file
model.save("unet.h5")