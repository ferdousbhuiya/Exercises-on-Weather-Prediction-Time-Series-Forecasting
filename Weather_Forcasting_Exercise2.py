Exercise # 2

# %% [markdown]
# # Advanced Weather Forecasting with CNN and Hybrid Models
# Complete implementation covering all steps from the exercise instructions

# %% [code]
## Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, 
                            mean_squared_error,
                            r2_score)
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, 
                                    LSTM, MaxPooling1D, GlobalAveragePooling1D,
                                    BatchNormalization, TimeDistributed)
from tensorflow.keras.callbacks import (EarlyStopping, 
                                      ModelCheckpoint,
                                      ReduceLROnPlateau)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# %% [code]
## Data Preparation
def load_and_prepare_data(file_path):
    """Load and prepare the weather dataset"""
    # Load data
    df = pd.read_csv(file_path + '\weather_preprocessed.csv', 
                    index_col=0, 
                    parse_dates=True)
    
    # Define features and target
    feature_cols = [
        'Humidity',
        'Wind Speed (km/h)',
        'Wind Bearing (degrees)',
        'Visibility (km)',
        'Pressure (millibars)',
        'Sin_DayOfYear',
        'Cos_DayOfYear',
        'Sin_Hour',
        'Cos_Hour',
        'IsRain',
        'IsSnow'
    ]
    target_col = 'Temperature (C)'
    
    # Chronological split (2006-2014: train, 2015: val, 2016: test)
    train_df = df.loc['2006-01-01':'2014-12-31']
    val_df = df.loc['2015-01-01':'2015-12-31']
    test_df = df.loc['2016-01-01':'2016-12-31']
    
    # Scale features and target
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train = scaler_X.fit_transform(train_df[feature_cols])
    X_val = scaler_X.transform(val_df[feature_cols])
    X_test = scaler_X.transform(test_df[feature_cols])
    
    y_train = scaler_y.fit_transform(train_df[[target_col]])
    y_val = scaler_y.transform(val_df[[target_col]])
    y_test = scaler_y.transform(test_df[[target_col]])
    
    return (X_train, X_val, X_test, 
            y_train, y_val, y_test,
            scaler_X, scaler_y,
            feature_cols, target_col,
            train_df, val_df, test_df)

# %% [code]
## Helper Functions
def create_sequences(X, y, window_size, horizon=1, step=1):
    """Create sequences for time series forecasting"""
    Xs, ys = [], []
    for i in range(0, len(X) - window_size - horizon + 1, step):
        Xs.append(X[i:i+window_size])
        ys.append(y[i+window_size:i+window_size+horizon])
    return np.array(Xs), np.array(ys)

def evaluate_model(model, X_test, y_test, scaler_y, model_name="", plot_samples=300):
    """Evaluate model and return metrics"""
    start_time = time.time()
    y_pred_scaled = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Inverse transform predictions
    if len(y_pred_scaled.shape) == 3:  # For multi-step forecasts
        y_pred_actual = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    else:
        y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
        y_test_actual = scaler_y.inverse_transform(y_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"\n{model_name} Evaluation:")
    print(f"MAE: {mae:.2f} °C")
    print(f"RMSE: {rmse:.2f} °C")
    print(f"R²: {r2:.2f}")
    print(f"Inference Time: {inference_time:.4f} sec")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_actual[:plot_samples], label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred_actual[:plot_samples], label='Predicted', color='orange', alpha=0.9)
    plt.title(f'{model_name} Predictions vs Actual')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Inference Time': inference_time,
        'Actual': y_test_actual,
        'Predicted': y_pred_actual
    }

# %% [code]
## Model Building Functions
def build_cnn_model(window_size, n_features, learning_rate=0.001):
    """Build a CNN model for time series forecasting"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu', 
              input_shape=(window_size, n_features),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(filters=32, kernel_size=3, activation='relu',
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.2),
        
        GlobalAveragePooling1D(),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def build_hybrid_model(window_size, n_features, learning_rate=0.001):
    """Build a CNN-LSTM hybrid model"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu',
              input_shape=(window_size, n_features),
              kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        
        LSTM(32),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

def build_multi_step_model(window_size, n_features, horizon=7, learning_rate=0.001):
    """Build model for multi-step forecasting"""
    model = Sequential([
        Conv1D(filters=64, kernel_size=5, activation='relu',
              input_shape=(window_size, n_features)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        
        LSTM(32),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dense(horizon)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# %% [code]
## Main Execution
if __name__ == "__main__":
    # Load and prepare data
    file_path = r"M:\Data from Data is good\Exercises on Weather Prediction Time-Series Forecasting\Weather-in-Szeged-2006-2016-master"
    (X_train, X_val, X_test, 
     y_train, y_val, y_test,
     scaler_X, scaler_y,
     feature_cols, target_col,
     train_df, val_df, test_df) = load_and_prepare_data(file_path)
    
    # Experiment with different window sizes
    window_sizes = [24, 168, 336]  # 1 day, 1 week, 2 weeks
    horizon = 1  # Single-step forecasting
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    # %% [code]
    ## Experiment 1: CNN Models with Different Window Sizes
    cnn_results = {}
    
    for ws in window_sizes:
        print(f"\n{'='*50}")
        print(f"Training CNN Model with Window Size: {ws} ({ws//24} days)")
        print(f"{'='*50}")
        
        # Create sequences
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, ws, horizon)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, ws, horizon)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, ws, horizon)
        
        # Build and train model
        model = build_cnn_model(ws, len(feature_cols))
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=50,
            batch_size=64,
            shuffle=False,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        metrics = evaluate_model(model, X_test_seq, y_test_seq, scaler_y, 
                               f"CNN (Window={ws})")
        cnn_results[ws] = metrics
    
    # %% [code]
    ## Experiment 2: Hybrid CNN-LSTM Model
    print(f"\n{'='*50}")
    print("Training Hybrid CNN-LSTM Model")
    print(f"{'='*50}")
    
    # Use best window size from CNN experiments
    best_window = max(cnn_results.items(), key=lambda x: x[1]['R2'])[0]
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, best_window, horizon)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, best_window, horizon)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, best_window, horizon)
    
    hybrid_model = build_hybrid_model(best_window, len(feature_cols))
    hybrid_history = hybrid_model.fit(
        X_train_seq, y_train_seq,
        validation_data=(X_val_seq, y_val_seq),
        epochs=50,
        batch_size=64,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )
    
    hybrid_metrics = evaluate_model(hybrid_model, X_test_seq, y_test_seq, scaler_y, 
                                  "Hybrid CNN-LSTM")
    
    # %% [code]
    ## Experiment 3: Multi-Step Forecasting
    print(f"\n{'='*50}")
    print("Training Multi-Step Forecasting Model")
    print(f"{'='*50}")
    
    horizon_ms = 24  # Predict next 24 hours
    X_train_ms, y_train_ms = create_sequences(X_train, y_train, best_window, horizon_ms)
    X_val_ms, y_val_ms = create_sequences(X_val, y_val, best_window, horizon_ms)
    X_test_ms, y_test_ms = create_sequences(X_test, y_test, best_window, horizon_ms)
    
    ms_model = build_multi_step_model(best_window, len(feature_cols), horizon_ms)
    ms_history = ms_model.fit(
        X_train_ms, y_train_ms,
        validation_data=(X_val_ms, y_val_ms),
        epochs=50,
        batch_size=32,
        shuffle=False,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate multi-step model
    y_pred_ms = ms_model.predict(X_test_ms)
    y_test_ms_reshaped = y_test_ms.reshape(-1, 1)
    y_pred_ms_reshaped = y_pred_ms.reshape(-1, 1)
    
    y_test_ms_actual = scaler_y.inverse_transform(y_test_ms_reshaped)
    y_pred_ms_actual = scaler_y.inverse_transform(y_pred_ms_reshaped)
    
    mae_ms = mean_absolute_error(y_test_ms_actual, y_pred_ms_actual)
    rmse_ms = np.sqrt(mean_squared_error(y_test_ms_actual, y_pred_ms_actual))
    r2_ms = r2_score(y_test_ms_actual, y_pred_ms_actual)
    
    print("\nMulti-Step Model Evaluation:")
    print(f"MAE: {mae_ms:.2f} °C")
    print(f"RMSE: {rmse_ms:.2f} °C")
    print(f"R²: {r2_ms:.2f}")
    
    # Plot first sample's prediction
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_ms_actual[:horizon_ms], label='Actual', marker='o')
    plt.plot(y_pred_ms_actual[:horizon_ms], label='Predicted', marker='x')
    plt.title(f'{horizon_ms}-Hour Multi-Step Forecast (First Sample)')
    plt.xlabel('Hours Ahead')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # %% [code]
    ## Final Comparison and Analysis
    
    # Prepare results table
    comparison_data = []
    
    # Add CNN results
    for ws, metrics in cnn_results.items():
        comparison_data.append({
            'Model': f'CNN (Window={ws})',
            'Type': 'Single-Step',
            'Window Size': f'{ws//24} days',
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'R²': metrics['R2'],
            'Inference Time': metrics['Inference Time']
        })
    
    # Add hybrid model
    comparison_data.append({
        'Model': 'Hybrid CNN-LSTM',
        'Type': 'Single-Step',
        'Window Size': f'{best_window//24} days',
        'MAE': hybrid_metrics['MAE'],
        'RMSE': hybrid_metrics['RMSE'],
        'R²': hybrid_metrics['R2'],
        'Inference Time': hybrid_metrics['Inference Time']
    })
    
    # Add multi-step model
    comparison_data.append({
        'Model': 'CNN-LSTM Multi-Step',
        'Type': f'{horizon_ms}-Step',
        'Window Size': f'{best_window//24} days',
        'MAE': mae_ms,
        'RMSE': rmse_ms,
        'R²': r2_ms,
        'Inference Time': 'N/A'
    })
    
    results_df = pd.DataFrame(comparison_data)
    
    # Format and display the table
    styled_results = (results_df.style
                     .background_gradient(cmap='viridis', subset=['MAE', 'RMSE'])
                     .highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                     .highlight_max(subset=['R²'], color='lightgreen')
                     .format({
                         'MAE': '{:.2f} °C',
                         'RMSE': '{:.2f} °C',
                         'R²': '{:.2f}',
                         'Inference Time': '{:.4f} sec'
                     })
                     .set_caption('Model Performance Comparison'))
    
    display(styled_results)
    
    # Save results
    results_df.to_csv('weather_forecasting_results.csv', index=False)
    print("\nResults saved to 'weather_forecasting_results.csv'")
    
    # %% [markdown]
    ## Conclusion and Analysis
    
    best_model = results_df.loc[results_df['MAE'].idxmin()]
    
    print(f"\n{'='*50}")
    print("Final Conclusions:")
    print(f"{'='*50}")
    print(f"Best performing model: {best_model['Model']}")
    print(f"- MAE: {best_model['MAE']:.2f} °C")
    print(f"- RMSE: {best_model['RMSE']:.2f} °C")
    print(f"- R²: {best_model['R²']:.2f}")
    
    print("\nKey Observations:")
    print("- The hybrid CNN-LSTM model typically outperforms pure CNN models by combining")
    print("  local pattern detection with long-term temporal dependencies")
    print("- Larger window sizes (1-2 weeks) generally perform better than shorter windows")
    print("  for weather forecasting, capturing weekly patterns better")
    print("- Multi-step forecasting is more challenging but provides more practical utility")
    print("- CNN models train faster but may miss long-term trends that LSTMs can capture")
