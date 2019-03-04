import pandas as pd
import numpy as np
import create_training_set
from sklearn import preprocessing

import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model
from keras.callbacks import Callback, EarlyStopping

import seaborn as sns

import matplotlib.pyplot as plt


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error/ Points')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()
    fig.savefig("TrainingEpochsOne.pdf")

    fig_two = plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$points^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    fig_two.savefig("TrainingEpochesTwo.pdf")


class PrintDot(Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


if __name__ == "__main__":

    df_training = pd.read_csv("Data/training_set_features/2019-03-02/training_set_featurised.csv")

    df_training = df_training.drop("Unnamed: 0", axis = 1)

    n_full_training_set = len(df_training)

    min_minutes_per_game = 20

    df_training = df_training.assign(meets_min_minutes =
                                     df_training["total_minutes_played"] > min_minutes_per_game*df_training["round"])

    filtered_training_set = df_training[df_training["meets_min_minutes"] == True]

    filtered_training_set = filtered_training_set[filtered_training_set["played_next_game"] == True]

    filtered_training_set = filtered_training_set[filtered_training_set["played_last_3_games"] == True]

    n_filtered_training_set = len(filtered_training_set)

    features_to_use = ["assists", "goals_scored", "total_points", "goals_scored_per_minute", "total_points_per_minute",
                       "total_points_3_game_form","total_points_per_played_game","goals_scored_per_played_game",
                       "assists_per_played_game","goals_scored_per_minute","assists_per_minute","clean_sheets_per_played_game",
                       "value"]

    min_max_scaler = preprocessing.MinMaxScaler()

    features = filtered_training_set[features_to_use].values

    for feature in features_to_use:
        print(feature)
        pair_plot = sns.pairplot(filtered_training_set[[feature,"target"]], diag_kind="kde")
        pair_plot.savefig("pair_plot_{f}.pdf".format(f=feature))

    scaled_features= min_max_scaler.fit_transform(features)

    targets = filtered_training_set["target"].values

    split_boundary = int(len(scaled_features)/2)

    train_scaled_features = scaled_features[:split_boundary]

    test_scaled_features = scaled_features[split_boundary:]

    train_targets = targets[:split_boundary]

    test_targets = targets[split_boundary:]

    print(len(train_scaled_features))
    print(len(test_scaled_features))
    print(len(train_targets))
    print(len(test_targets))

    print(train_scaled_features.shape)

    Model = Sequential()
    Model.add(Dense(512, activation='relu', input_shape=[train_scaled_features.shape[1]]))
    Model.add(Dense(128, activation='relu'))
    Model.add(Dense(1))

    optimizer = RMSprop(0.001)

    plot_model(Model, to_file='Model.png')

    Model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

    EPOCHS = 250

    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    history = Model.fit(train_scaled_features, train_targets, epochs=EPOCHS, callbacks=[early_stop, PrintDot()],
                        validation_split = 0.2)

    plot_history(history)

    Model.summary()

    test_predictions = Model.predict(test_scaled_features).flatten()

    train_predictions = Model.predict(train_scaled_features).flatten()

    print(test_targets)

    fig = plt.figure()
    ax = fig.gca()
    ax.cla()
    ax.scatter(test_targets, test_predictions)
    ax.set_xlabel("True Points")
    ax.set_ylabel("Predicted Points")
    ax.set_xlim(0.0, 25.0)
    ax.set_ylim(0.0, 25.0)
    ax.plot([0.0,25.0],[0.0,25.0])
    fig.savefig("PredVTrue_v1.pdf")

    errors = test_predictions - test_targets
    print(errors)

    error_fig = plt.figure()
    ax_error = error_fig.gca()
    ax_error.cla()
    ax_error.hist(errors, bins =10)
    ax_error.set_xlabel("Prediction Error")
    error_fig.savefig("PredictionErrors.pdf")

    print(" Done error plot")
    #loss, mae, mse = Model.evaluate(test_scaled_features, test_targets, verbose=0)

    #print("Testing set Mean Abs Error: {:5.2f} Points".format(mae))


    predictions_fig = plt.figure("predictions")
    ax_predictions = predictions_fig.gca()
    ax_predictions.cla()
    ax_predictions.hist(test_predictions, bins = 10, color = "skyblue", alpha=0.5, label = "test",normed=True)
    ax_predictions.hist(train_predictions, bins = 10, color = "red", alpha=0.5, label= "train",normed=True)
    ax_predictions.hist(np.append(test_targets, train_targets), bins = 10,
                        color = "green", alpha=0.5, label= "actual",normed=True)
    ax_predictions.set_xlabel("Points")
    ax_predictions.legend(loc='best')
    predictions_fig.savefig("PredictionsDistributions.pdf")

    print("Done")