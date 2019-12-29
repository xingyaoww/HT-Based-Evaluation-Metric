#!/anaconda3/bin/python



if __name__ == '__main__':
    from keras.callbacks import ModelCheckpoint, CSVLogger

    # 0. Time Tracker
    import datetime
    starttime = datetime.datetime.now()
    def getTime(starttime=starttime):
        endtime = datetime.datetime.now()
        return (endtime - starttime).seconds

    while True:
        try:
            Trainning_Limit = input('> Load Dataset - image number [MAX 88880]:')
            # Trainning_Limit = ''
            if len(Trainning_Limit) < 1:
                Trainning_Limit = 88880
                break
            else:
                Trainning_Limit = int(Trainning_Limit)
                break
        except:
            print('The number should be INTEGER. TRY AGAIN')

    # Trainning_Limit = 88800
    print(f"* Loading {Trainning_Limit} Samples ")


    # IMPORT MODEL
    from buildmodel import *
    # IMPORT DATA GENERATOR
    import loadDataset
    # The GPU will crash under batch_size=16
    training_generator = loadDataset.DataGenerator(loadDataset.getTrainDataPath(TRAINING_LIMIT=Trainning_Limit),
                                                   batch_size=8)
    validation_generator = loadDataset.DataGenerator(loadDataset.getValidateDataPath())

    # Load Previous Weight
    # model.load_weights('2ndTrain_201903027/Lane_MOdel_U_GCN_OnlyWeights.h5', by_name=True)


    # Model Checkpoints
    checkpoint = ModelCheckpoint('weights.Epoch{epoch:02d}-loss{loss:.2f}-binary_loss:{binary_crossentropy:.5f}.hdf5',
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True
                                 )
    csv_logger = CSVLogger('training.log')
    callback_list = [checkpoint, csv_logger]

    # Model Train
    print(f'[{getTime()}s] Trainning Start')
    history = model.fit_generator(training_generator,
                                  # steps_per_epoch=5555,
                                  epochs=100,
                                  # verbose=2,
                                  max_queue_size=2,
                                  validation_data=validation_generator,
                                  validation_steps=3,
                                  workers=4,
                                  use_multiprocessing=True,
                                  # callbacks=callback_list
                                  )



    # Model Save
    print(f'[{getTime()}s] Trainning Completed')
    print(f'[{getTime()}s] Saving Model')

    model.save('Model_GCN.h5')
    model_yaml = model.to_yaml()
    with open("Model_GCN_OnlyModel.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    model.save_weights('Model_GCN_OnlyWeights.h5')
