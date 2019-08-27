from util import *
from segnet2 import segnet
from keras.callbacks import EarlyStopping,ModelCheckpoint
import h5py

trainimg_dir = "/home/maja/Projects/IPIRI/data_png/train/images/"
trainmsk_dir = "/home/maja/Projects/IPIRI/data_png/train/labels/"
valimg_dir = "/home/maja/Projects/IPIRI/data_png/test/images/"
valmsk_dir = "/home/maja/Projects/IPIRI/data_png/test/labels/"

classes = 3
inshape = (512,512) #orig 1500x1500 => 500x500
input_shape = (512, 512,3)
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




#weights from pretrained segnet model
weights = h5py.File('camvid_model_weights.hdf5')

model = segnet(input_shape=(512,512,3),nb_classes=3)

#current layers
model_layers = dict([(layer.name,layer) for layer in model.layers])
layer_names = [layer.name for layer in model.layers]
ii=0
for i in model_layers.keys():
    weight_names = weights['model_weights'][i].attrs["weight_names"]
    w = [weights['model_weights'][i][j] for j in weight_names]
    idx = layer_names.index(i)
    try:
        model.layers[idx].set_weights(w)
        print("layer: "+str(ii) + " loaded")
        ii=ii+1
    except Exception:
        print("layer: "+str(ii)+" not loaded")
        ii=ii+1
        continue

for i in range (0,len(model.layers)-4):
    model.layers[i].trainable = False

mcp = ModelCheckpoint('finetuned_model_weights.hdf5',save_best_only=True,monitor='val_loss',mode='min')

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit_generator(train_gen,epochs=10,callbacks=[mcp],validation_data=val_gen,validation_steps=10,steps_per_epoch=50)
