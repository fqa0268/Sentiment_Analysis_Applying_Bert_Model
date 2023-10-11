import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

def PlotTrain( history, model_name="???"):
    fig, axs = plt.subplots( 1, 2, figsize=(12, 5) )

    # Determine the name of the key that indexes into the accuracy metric
    acc_string = 'accuracy'

    # Plot loss
    axs[0].plot(history.history['loss'])
    axs[0].plot(history.history['val_loss'])
    axs[0].set_title(model_name + " " + 'model loss')
    axs[0].set_ylabel('loss')
    axs[0].set_xlabel('epoch')
    axs[0].legend(['train', 'validation'], loc='upper left')

    # Plot accuracy
    axs[1].plot(history.history[ acc_string ])
    axs[1].plot(history.history['val_' + acc_string ])
    axs[1].set_title(model_name + " " +'model accuracy')
    axs[1].set_ylabel('accuracy')
    axs[1].set_xlabel('epoch')
    axs[1].legend(['train', 'validation'], loc='upper left')

    return fig, axs


def TrainModel(model, data_train, data_val, data_test, n_batch=32, num_epochs = 5):
    num_train_steps = len(data_train[1]) * num_epochs
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps
    )

    model.compile(optimizer=Adam(learning_rate=lr_scheduler), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                    )
    
    history = model.fit(
        x=data_train[0],
        y=data_train[1],
        validation_data=(data_val[0], data_val[1]),
        batch_size=n_batch,
        callbacks=[early_stopping],
        epochs=num_epochs,
        verbose=0
    )

    PlotTrain(history, model.name)

    train_score = model.evaluate(data_train[0], data_train[1], verbose=0)
    val_score = model.evaluate(data_val[0], data_val[1], verbose=0)
    test_score = model.evaluate(data_test[0], data_test[1], verbose=0)
    score = {'train_loss': train_score[0], 'train_accuracy': train_score[1],
             'val_loss': val_score[0], 'val_accuracy': val_score[1],
             'test_loss': test_score[0], 'test_accuracy': test_score[1]}
    return model, score
    
    
def TrainModelTFDS(model, data_train, data_val, data_test, n_batch=32, num_epochs=10):
    size = data_train.reduce(0, lambda x, _: x + 1).numpy()
    num_train_steps = size * num_epochs
    lr_scheduler = PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps)

    model.compile(optimizer=Adam(learning_rate=lr_scheduler), loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    
    early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                    )
    
    history = model.fit(
        data_train.batch(n_batch),
        validation_data=data_val.batch(n_batch),
        callbacks=[early_stopping],
        epochs=num_epochs,
        verbose=0
        )

    PlotTrain(history, model.name)

    train_score = model.evaluate(data_train.batch(n_batch), verbose=0)
    val_score = model.evaluate(data_val.batch(n_batch), verbose=0)
    test_score = model.evaluate(data_test.batch(n_batch), verbose=0)
    score = {'train_loss': train_score[0], 'train_accuracy': train_score[1],
             'val_loss': val_score[0], 'val_accuracy': val_score[1],
             'test_loss': test_score[0], 'test_accuracy': test_score[1]}

    return model, score


def EvaluateModel(model, X, y_true = None):
    try:
        y_logits = model.predict(X.batch(32), verbose=0)
        y_true = np.array([y for x, y in X])
    except:
        y_logits = model.predict(X, verbose=0)
    
    try:
        y_logits = y_logits.logits
    except:
        pass

    y_pred = np.argmax(y_logits, axis=1)
    classification_summary = classification_report(y_true, y_pred, digits=3, zero_division=0)
    print(classification_summary)

    confusion_mat = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.heatmap(confusion_mat,
                annot=True,
                fmt='d',
                cmap='Greens',
                ax=ax);

    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5, 2.5], ['negative','neutral','positive'] , fontsize=9)
    plt.ylabel('True Label')
    plt.yticks([0.5, 1.5, 2.5], ['negative','neutral','positive'] , fontsize=9);


def ModelPath(model_name, fine_method, sub_dataset, train_size, ds_bool):
    sep = '_'
    file_name = sep.join([model_name.replace('/', '_'), fine_method, sub_dataset, str(train_size), ds_bool])
    return file_name


def SaveModel(model, model_name, fine_method, sub_dataset, train_size, ds_bool):
    file_name = ModelPath(model_name, fine_method, sub_dataset, train_size, ds_bool)
    model.save_weights('./Model/' + file_name)


def LoadModel(model_name, fine_method, sub_dataset, train_size, ds_bool):
    if model_name == 'MyCustomModel':
        model = MyCustomModel()
    else:
        model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    file_name = ModelPath(model_name, fine_method, sub_dataset, train_size, ds_bool)
    model.load_weights('./Model/' + file_name)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model
    
    
summary_table = pd.DataFrame(columns = ['Model', 'Fine Method', 'Sub-Dataset', 'Train Size', 'tf Dataset', 'train loss', 'train acc', 'val loss', 'val acc', 'test loss', 'test acc'])
def RecordToSummary(summary_table, model_name, fine_method, sub_dataset, train_size, ds_bool, score):
    """
    Record a result in a table for convenient comparison.
    """
    row = pd.DataFrame({'Model':model_name, 'Fine Method': fine_method,
                        'Sub-Dataset': sub_dataset, 'Train Size': train_size, 'tf Dataset': ds_bool,
                        'train loss': score['train_loss'], 'train acc': score['train_accuracy'],
                        'val loss': score['val_loss'], 'val acc': score['val_accuracy'],
                        'test loss': score['test_loss'], 'test acc': score['test_accuracy']}, index = [0])
    summary_table = pd.concat([summary_table, row], ignore_index = True)
    return summary_table
