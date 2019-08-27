from util import *
from segnet2 import segnet
from keras.callbacks import EarlyStopping,ModelCheckpoint

trainimg_dir = "/home/maja/Projects/IPIRI/CamVid/train/"
trainmsk_dir = "/home/maja/Projects/IPIRI/CamVid/trainannot/"
valimg_dir = "/home/maja/Projects/IPIRI/CamVid/test/"
valmsk_dir = "/home/maja/Projects/IPIRI/CamVid/testannot/"

classes = 12
inshape = (360,480)
input_shape = (360, 480,3)
batch = 5

train_gen = data_gen_small( trainimg_dir,
                             trainmsk_dir,
                             batch,
                             inshape, 
                             classes)
val_gen = data_gen_small(valimg_dir, 
                         valmsk_dir,
                         batch,
                         inshape,
                         classes)


model = segnet(input_shape=input_shape)
mcp = ModelCheckpoint('camvid_model_weights.hdf5',save_best_only=True,monitor='val_loss',mode='min')

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit_generator(train_gen,epochs=10,callbacks=[mcp],validation_data=val_gen,validation_steps=10,steps_per_epoch=100)
