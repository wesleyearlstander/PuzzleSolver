from UNet import *
from DataAugmentation import *

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

def TrainModelAugmentation(images,masks,aug_list, learning_rate=5e-5): #save checkpoint of best model
    imageFolds, maskFolds = K_fold(images,masks,6) #final run on one fold
    trainingImages = []
    validationMasks = []
    validationImages = []
    trainingMasks = []
    model = UNet(imgs[0], learning_rate)
    trainingImages = np.array(imageFolds[:4]).reshape(-1, 192, 192, 3)
    validationImages = np.array(imageFolds[4]).reshape(-1, 192, 192, 3)
    testImages = np.array(imageFolds[5]).reshape(-1, 192, 192, 3)
    validationMasks = np.array(maskFolds[4]).reshape(-1, 192, 192, 1)
    trainingMasks = np.array(maskFolds[:4]).reshape(-1, 192, 192, 1)
    testMasks = np.array(maskFolds[5]).reshape(-1, 192, 192, 1)
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
    training_gen = trainGenerator(trainingImages,trainingMasks,aug_list)
    train_gen = (pair for pair in training_gen)
    model.UNet.fit(train_gen, validation_data=(validationImages, validationMasks), steps_per_epoch=150, epochs=200,verbose=1, shuffle=True, callbacks=[model_checkpoint]) #WandbCallback()
    model.UNet.load_weights("unet.hdf5")
    results = model.UNet.evaluate(testImages, testMasks, batch_size=1)
    del model.UNet #  to avoid any trace of model
    del model
    tf.compat.v1.reset_default_graph()
    clear_session() # removing session, it will instance another


data_dict = dict(rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2, 
                fill_mode="nearest")

TrainModelAugmentation(imgs, msks, data_dict) #huge variation