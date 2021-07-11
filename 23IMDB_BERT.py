# install ktrain
!pip3 install ktrain

# import ktrain
import ktrain
from ktrain import text

# load training and validation data from a folder
DATADIR = 'data/aclImdb'
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_folder(DATADIR, 
                                                                       maxlen=500, 
                                                                       preprocess_mode='bert',
                                                                       train_test_names=['train', 
                                                                                         'test'],
                                                                       classes=['pos', 'neg'])

model = text.text_classifier('bert', (x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model,train_data=(x_train, y_train), val_data=(x_test, y_test), batch_size=6)

learner.lr_find()
learner.lr_plot()

learner.fit(0.001, 3, cycle_len=1, cycle_mult=2)

predictor = ktrain.get_predictor(learner.model, preproc)

data = [ 'This movie was horrible! The plot was boring. Acting was okay, though.',
         'The film really sucked. I want my money back.',
        'What a beautiful romantic comedy. 10/10 would see again!']

predictor.predict(data)
