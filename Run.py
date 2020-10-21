from UNet import *
from DataAugmentation import *
#scikit
# import wandb
# from wandb.keras import WandbCallback

if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# wandb.init(project="unet")


def validateModel():
    imageFolds, maskFolds = K_fold(imgs,msks,6)
    accuracies = []
    for i in range(6):
        model = UNet(imgs[0])
        trainingImages = np.array(imageFolds[:i] + imageFolds[i+1:]).reshape(-1, 192, 192, 3)
        testImages = np.array(imageFolds[i]).reshape(-1, 192, 192, 3)
        trainingMasks = np.array(maskFolds[:i] + maskFolds[i+1:]).reshape(-1, 192, 192, 1)
        testMasks = np.array(maskFolds[i]).reshape(-1, 192, 192, 1)

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
        model.UNet.fit(trainingImages, trainingMasks, batch_size=2, epochs=1,verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint]) #WandbCallback()
        results = model.UNet.evaluate(testImages, testMasks, batch_size=1)
        accuracies.append(results[1])
        print("test loss:", results[0], ", test accuracy:", results[1])
    print(np.mean(accuracies))

validateModel()
# batch size = 1 causes bad gradient on outliers, full step size in the outlier reducing the full overall performance. 
# batches provide a smoothing on gradient steps
# need test set
# 6 cross fold validation ( and get average ) 
# data-augmentation to increase data size (flip, rotation, affine transform, random cropping, keras.preprocessing)
# pre-cleaning data, and post-cleaning data
# wandb and move network onto GPU