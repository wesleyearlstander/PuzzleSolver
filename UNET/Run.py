from UNet import *
from DataAugmentation import *
import logging
from sklearn.metrics import roc_auc_score
#scikit
# import wandb
# from wandb.keras import WandbCallback

logging.basicConfig(filename="learning_rates.log", level=logging.DEBUG)

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# wandb.init(project="unet")

logging.debug(str(100))

def validateModel(images,masks,name,learning_rate=1e-4):
    imageFolds, maskFolds = K_fold(images,masks,6)
    accuracies = []
    trainingImages = []
    validationMasks = []
    validationImages = []
    trainingMasks = []
    total_scores = []
    for i in range(6):
        model = UNet(imgs[0], learning_rate)
        if i+1 < 6:
            if i+2 < 6:
                trainingImages = np.array(imageFolds[:i] + imageFolds[i+2:]).reshape(-1, 192, 192, 3)
            else:
                trainingImages = np.array(imageFolds[:i]).reshape(-1, 192, 192, 3)
            validationImages = np.array(imageFolds[i+1]).reshape(-1, 192, 192, 3)
        else:
            trainingImages = np.array(imageFolds[1:i]).reshape(-1, 192, 192, 3)
            validationImages = np.array(imageFolds[0]).reshape(-1, 192, 192, 3)
        testImages = np.array(imageFolds[i]).reshape(-1, 192, 192, 3)
        if i+1 < 6:
            validationMasks = np.array(maskFolds[i+1]).reshape(-1, 192, 192, 1)
            if i+2 < 6:
                trainingMasks = np.array(maskFolds[:i] + maskFolds[i+2:]).reshape(-1, 192, 192, 1)
            else:
                trainingMasks = np.array(maskFolds[:i]).reshape(-1, 192, 192, 1)
        else:
            validationMasks = np.array(maskFolds[0]).reshape(-1, 192, 192, 1)
            trainingMasks = np.array(maskFolds[1:i]).reshape(-1, 192, 192, 1)
        testMasks = np.array(maskFolds[i]).reshape(-1, 192, 192, 1)

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
        model.UNet.fit(trainingImages, trainingMasks, validation_data=(validationImages, validationMasks), batch_size=2, epochs=200,verbose=1, shuffle=True, callbacks=[model_checkpoint, es]) #WandbCallback()
        model.UNet.load_weights("unet.hdf5")
        results = model.UNet.evaluate(testImages, testMasks, batch_size=1)
        accuracies.append(results[1])
        ypred = model.UNet.predict(testImages.reshape(-1,192,192,3))
	    #ypred = (ypred - np.max(ypred))/(np.max(ypred)-np.min(ypred))
        ypred = np.uint8(ypred > 0.5)
        score = roc_auc_score(ypred.flatten(), testMasks.flatten())
        total_scores.append(score)
        logging.debug(name + " -- test loss:" + str(results[0]) + ", test accuracy:" + str(results[1]))
        del model.UNet #  for avoid any trace on aigen
        del model
        tf.compat.v1.reset_default_graph() # for being sure
        clear_session() # removing session, it will instance another
    logging.debug(str(learning_rate) + " " + name + " - ave_accuracy:" + str(np.mean(accuracies)) + " - ROC " + str(np.mean(total_scores)))

def validateModelAugmentation(images,masks,aug_list, name, learning_rate=1e-4):
    imageFolds, maskFolds = K_fold(images,masks,6)
    accuracies = []
    trainingImages = []
    validationMasks = []
    validationImages = []
    trainingMasks = []
    total_scores = []
    for i in range(6):
        model = UNet(imgs[0], learning_rate)
        if i+1 < 6:
            if i+2 < 6:
                trainingImages = np.array(imageFolds[:i] + imageFolds[i+2:]).reshape(-1, 192, 192, 3)
            else:
                trainingImages = np.array(imageFolds[:i]).reshape(-1, 192, 192, 3)
            validationImages = np.array(imageFolds[i+1]).reshape(-1, 192, 192, 3)
        else:
            trainingImages = np.array(imageFolds[1:i]).reshape(-1, 192, 192, 3)
            validationImages = np.array(imageFolds[0]).reshape(-1, 192, 192, 3)
        testImages = np.array(imageFolds[i]).reshape(-1, 192, 192, 3)
        if i+1 < 6:
            validationMasks = np.array(maskFolds[i+1]).reshape(-1, 192, 192, 1)
            if i+2 < 6:
                trainingMasks = np.array(maskFolds[:i] + maskFolds[i+2:]).reshape(-1, 192, 192, 1)
            else:
                trainingMasks = np.array(maskFolds[:i]).reshape(-1, 192, 192, 1)
        else:
            validationMasks = np.array(maskFolds[0]).reshape(-1, 192, 192, 1)
            trainingMasks = np.array(maskFolds[1:i]).reshape(-1, 192, 192, 1)
        testMasks = np.array(maskFolds[i]).reshape(-1, 192, 192, 1)
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
        training_gen = trainGenerator(trainingImages,trainingMasks,aug_list)
        train_gen = (pair for pair in training_gen)
        model.UNet.fit(train_gen, validation_data=(validationImages, validationMasks), steps_per_epoch=150, epochs=200,verbose=1, shuffle=True, callbacks=[model_checkpoint, es]) #WandbCallback()
        model.UNet.load_weights("unet.hdf5")
        results = model.UNet.evaluate(testImages, testMasks, batch_size=1)
        ypred = model.UNet.predict(testImages.reshape(-1,192,192,3))
	    #ypred = (ypred - np.max(ypred))/(np.max(ypred)-np.min(ypred))
        ypred = np.uint8(ypred > 0.5)
        score = roc_auc_score(ypred.flatten(), testMasks.flatten())
        total_scores.append(score)
        accuracies.append(results[1])
        logging.debug("augmented -- test loss:" + str(results[0]) + ", test accuracy:" + str(results[1]))
        del model.UNet #  for avoid any trace on model
        del model
        tf.compat.v1.reset_default_graph() # for being sure
        clear_session() # removing session, it will instance another
    logging.debug(str(learning_rate) + " augmented " + name + " - ave_accuracy:" + str(np.mean(accuracies)) + " - ROC " + str(np.mean(total_scores)))

data_dict = dict(rotation_range=45,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2, 
                fill_mode="nearest")

data_dict2 = dict(rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.1, 
                fill_mode="nearest")

learning_rates = [1e-4, 1e-5, 1e-6, 5e-5, 5e-6, 5e-4]
for i in range(len(learning_rates)):
    validateModel(imgs,msks, "default model", learning_rates[i]) #default model
    validateModel(imgs,msksCleaned, "cleaned masks", learning_rates[i]) #cleaned masks
    validateModelAugmentation(imgs, msks, data_dict, "intense", learning_rates[i]) #huge variation
    validateModelAugmentation(imgs, msks, data_dict2, "mild", learning_rates[i]) #mild variation






# batch size = 1 causes bad gradient on outliers, full step size in the outlier reducing the full overall performance. 
# batches provide a smoothing on gradient steps
# need test set
# 6 cross fold validation ( and get average ) 
# data-augmentation to increase data size (flip, rotation, affine transform, random cropping, keras.preprocessing)
# pre-cleaning data, and post-cleaning data
# wandb and move network onto GPU