import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam
import os

# XGBOOST TRAINER
# DATASET DIR
def load_data(folder):
    file_path = f"dataset/{folder}/{folder.lower()}_sales.csv"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    data = pd.read_csv(file_path)

    data.columns = data.columns.str.strip().str.lower()

    if 'total sales' in data.columns:
        data.rename(columns={'total sales': 'sales'}, inplace=True)
    if 'branch' not in data.columns:
        data['branch'] = None
    if 'size' not in data.columns:
        data['size'] = None

    # FEATURE ENGINEERING (FOR IMPROVEMENTS & ACCURACY)
    data['date'] = pd.to_datetime(data['date'])
    data['dayofweek'] = data['date'].dt.dayofweek
    data['month'] = data['date'].dt.month
    data['dayofyear'] = data['date'].dt.dayofyear
    data['is_weekend'] = data['dayofweek'].isin([5, 6])

    data = data.drop(columns=['date'])

    # LOG TRANSFORMATION (FOR IMPROVEMENTS & ACCURACY)
    data['sales'] = data['sales'].apply(lambda x: np.log(x + 1))

    # CYCLICAL ENCODING (FOR IMPROVEMENTS & ACCURACY)
    data['dayofweek_sin'] = np.sin(2 * np.pi * data['dayofweek']/7)
    data['dayofweek_cos'] = np.cos(2 * np.pi * data['dayofweek']/7)
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)

    return data

# DATASETS LOADER
kkopi_data = load_data('Kkopi')
waffly_data = load_data('Waffly')
water_station_data = load_data('WaterStation')

if kkopi_data is None or waffly_data is None or water_station_data is None:
    print("Error: One or more datasets could not be loaded.")
else:
    data = pd.concat([kkopi_data, waffly_data, water_station_data], ignore_index=True)
    data = pd.get_dummies(data, columns=['branch', 'product', 'size'], drop_first=True)
    X = data.drop(columns=['sales'])
    y = data['sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # CREATE MODEL (W/ HYPERPARAMETERS)
    def create_model(learning_rate=0.001, dropout_rate=0.2, l2_reg=0.01):
        model = Sequential()
        model.add(Dense(256, input_dim=X_train_scaled.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg)))
        model.add(Dense(1, activation='linear'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    # HYPERPARAMETER TUNING (FOR IMPROVEMENTS & ACCURACY)
    best_rmse = float('inf')
    best_model = None
    best_params = {}

    learning_rates = [0.001, 0.01, 0.0001]
    dropout_rates = [0.2, 0.3, 0.4]
    l2_regs = [0.01, 0.001, 0.0001]

    for lr in learning_rates:
        for dr in dropout_rates:
            for reg in l2_regs:
                print(f"Training with LR={lr}, Dropout={dr}, L2 Regularization={reg}")
                model = create_model(learning_rate=lr, dropout_rate=dr, l2_reg=reg)

                # CHECKPOINT
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

                # TRAIN
                model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                          verbose=0, callbacks=[early_stopping, checkpoint])

                # EVALUATE
                loss = model.evaluate(X_test_scaled, y_test, verbose=0)
                y_pred = model.predict(X_test_scaled)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                print(f"Test RMSE: {rmse}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_params = {'learning_rate': lr, 'dropout_rate': dr, 'l2_reg': reg}

    print(f"Best Model RMSE: {best_rmse}")
    print(f"Best Hyperparameters: {best_params}")

    # SAVING
    best_model.save('sales_prediction_best_model.keras')
    print("model saved as 'sales_prediction_best_model.keras'")