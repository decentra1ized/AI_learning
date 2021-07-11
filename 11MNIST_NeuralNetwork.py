import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

loss_fig = Figure(figsize=(5,5), dpi=100)
loss_ax = loss_fig.add_subplot(111)
loss_ax.set_xlabel('Epochs')
loss_ax.set_ylabel('Loss')
loss_ax.set_title('History of loss')

acc_fig = Figure(figsize=(5,5), dpi=100)
acc_ax = acc_fig.add_subplot(111)
acc_ax.set_xlabel('Epochs')
acc_ax.set_ylabel('Accuracy')
acc_ax.set_title('History of accuracy')

digit_fig = Figure(figsize=(5,5), dpi=100)
digit_ax = digit_fig.add_subplot(111)
digit_ax.set_title('Selected Digit Image')
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 29, 5)
minor_ticks = np.arange(0, 29, 1)
digit_ax.set_xticks(major_ticks)
digit_ax.set_xticks(minor_ticks, minor=True)
digit_ax.set_yticks(major_ticks)
digit_ax.set_yticks(minor_ticks, minor=True)
# And a corresponding grid
digit_ax.grid(which='both')
# Or if you want different settings for the grids:
digit_ax.grid(which='minor', alpha=0.2)
digit_ax.grid(which='major', alpha=0.5)

number_fig = Figure(figsize=(5,5), dpi=100)
number_ax = number_fig.add_subplot(111)
number_ax.set_title('Probalility Chart')

def learing(): 
    """
    MAchine Learning, Keras 
    """
    # Keras Linear Regression
    import numpy as np
    from keras import optimizers
    from keras.layers import Dense
    from keras.models import Sequential
    
    t_a = int(t_aSpbox.get()) - 1
    t_t = int(t_tSpbox.get()) 
     
    selected_image = t_a
    
    # Keras Datasets Load from MNIST dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Try to find values for W and b to compute y_data = x_data * W + b
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(Dense(10, activation='softmax'))

    # placeholders for a tensor that will be always fed using feed_dict
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # Our hypothesis XW+b
    x_train = x_train.reshape((60000, 28 * 28))
    x_train = x_train.astype('float32') / 255

    x_test_original = x_test
    x_test = x_test.reshape((10000, 28 * 28))
    x_test = x_test.astype('float32') / 255

    # optimizer
    from keras.utils import to_categorical

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    """
    Callback EarlyStopping, ModelCheckpoint
    """
    # Import the early stopping callback
    from keras.callbacks import EarlyStopping, ModelCheckpoint

    # Save the best model as best_digits_model.hdf5
    modelCheckpoint = ModelCheckpoint('best_digits_model.hdf5', save_best_only=True)

    # Define a callback to monitor val_acc => val_accuracy
    monitor_val_acc = EarlyStopping(monitor='val_accuracy', 
                           patience=5)

    # Train the model using the early stopping callback
    history = model.fit(x_train, y_train, 
                        validation_data=(x_test, y_test),
                        epochs=t_t, batch_size=128,
                        callbacks=[monitor_val_acc, modelCheckpoint])

    result = model.predict(np.array([x_test[selected_image]]))
    result_number = np.argmax(result)

    loss_ax.plot(history.history['loss'], 'ro', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r:', label='val loss')
    # val_acc => val_accuracy
    acc_ax.plot(history.history['accuracy'], 'bo', label='train acc')
    acc_ax.plot(history.history['val_accuracy'], 'b:', label='val acc')
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='upper left')   
    
    digit_ax.imshow(x_test_original[selected_image])

    digits = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    y_pos = np.arange(len(digits))
    performance = [ val for val in result[0]]
    print(performance)
    result_probability = performance[result_number]
    number_ax.bar(y_pos, performance, align='center', alpha=0.5)
    number_ax.set_title('Number is %2i (probability %7.4f)' % (result_number, result_probability*100))

    loss_fig.canvas.draw()
    acc_fig.canvas.draw()
    digit_fig.canvas.draw()
    number_fig.canvas.draw()
     

#main
main = Tk()
main.title("MNIST Digits, Convolutional Neural Network")
main.geometry()

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=10000, increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=0,column=1)
t_aLabel=Label(main, text='Numer of Digits : ')                
t_aLabel.grid(row=0,column=0)

t_tVal  = IntVar(value=100)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=100, justify=RIGHT)
t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=0,column=3)
t_tLabel=Label(main, text='Number of trains : ')                
t_tLabel.grid(row=0,column=2)

Button(main,text="Deep Learing", height=1,command=lambda:learing()).grid(row=0, column=4, columnspan=2, sticky=(W, E))
    
loss_canvas = FigureCanvasTkAgg(loss_fig, main)
loss_canvas.get_tk_widget().grid(row=1,column=0,columnspan=3) 

acc_canvas = FigureCanvasTkAgg(acc_fig, main)
acc_canvas.get_tk_widget().grid(row=1,column=3,columnspan=3)

digit_canvas = FigureCanvasTkAgg(digit_fig, main)
digit_canvas.get_tk_widget().grid(row=2,column=0,columnspan=3) 

number_canvas = FigureCanvasTkAgg(number_fig, main)
number_canvas.get_tk_widget().grid(row=2,column=3,columnspan=3)

main.mainloop()