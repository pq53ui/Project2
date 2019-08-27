from util import *
from segnet2 import segnet
import h5py

img_dir = "/home/maja/MAG/IPIRI/project2/data_png/val/images/"
msk_dir = "/home/maja/MAG/IPIRI/project2/data_png/val/labels/"

classes = 3
inshape = (512,512) #orig 1500x1500 => 500x500
input_shape = (512, 512,3)
batch = 1


image,label = get_data_small(img_dir, 
                         msk_dir,
                         batch,
                         inshape,
                         classes)




#weights from pretrained segnet model
weights = h5py.File('./finetuned_model_weights.hdf5')

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


model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
model.fit_generator(train_gen,epochs=10,callbacks=[mcp],validation_data=val_gen,validation_steps=10,steps_per_epoch=50)
