"""
LSTM Neural Network for Stock Prediction.

Uses TensorFlow/Keras to build a recurrent neural network that learns
temporal patterns in stock returns and features.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List
import warnings
import logging

logger = logging.getLogger(__name__)

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    HAS_TF = True
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow not installed. LSTM model will not be available.")


class LSTMPredictor:
    """
    LSTM-based stock return predictor.
    
    This model uses a recurrent neural network to learn temporal patterns
    in stock features and predict future returns.
    
    Attributes:
        lookback: Number of past days to use for prediction (default 60)
        lstm_units: Number of LSTM units in each layer (default 50)
        dropout: Dropout rate for regularization (default 0.2)
        epochs: Maximum training epochs (default 100)
        batch_size: Training batch size (default 32)
        patience: Early stopping patience (default 10)
        
    Example:
        >>> predictor = LSTMPredictor(lookback=30, lstm_units=64)
        >>> predictor.fit(train_features_df, train_target_series)
        >>> predicted_return = predictor.predict(recent_features_df)
    """
    
    def __init__(
        self,
        lookback: int = 60,
        lstm_units: int = 50,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
        learning_rate: float = 0.001,
        n_layers: int = 2,
        use_attention: bool = False,
        verbose: int = 0,
        random_state: int = 42,
    ):
        """
        Initialize LSTM predictor.
        
        Args:
            lookback: Number of past time steps to use (window size)
            lstm_units: Number of units in LSTM layers
            dropout: Dropout rate for regularization
            epochs: Maximum training epochs
            batch_size: Batch size for training
            patience: Early stopping patience (epochs without improvement)
            learning_rate: Adam optimizer learning rate
            n_layers: Number of LSTM layers
            use_attention: Whether to use attention mechanism (experimental)
            verbose: Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)
            random_state: Random seed for reproducibility
        """
        if not HAS_TF:
            raise ImportError(
                "TensorFlow is required for LSTM model. "
                "Install with: pip install tensorflow"
            )
        
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.use_attention = use_attention
        self.verbose = verbose
        self.random_state = random_state
        
        self.model: Optional[keras.Model] = None
        self.scaler: Optional[StandardScaler] = None
        self.target_scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted: bool = False
        self.history: Optional[Dict] = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def _build_model(self, n_features: int) -> keras.Model:
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # Input layer with first LSTM
        model.add(Input(shape=(self.lookback, n_features)))
        
        # Stack LSTM layers
        for i in range(self.n_layers):
            return_sequences = (i < self.n_layers - 1)  # Only last layer returns single output
            model.add(LSTM(
                units=self.lstm_units,
                return_sequences=return_sequences,
                kernel_regularizer=keras.regularizers.l2(0.01),
            ))
            model.add(Dropout(self.dropout))
            if i < self.n_layers - 1:
                model.add(BatchNormalization())
        
        # Dense layers for final prediction
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(self.dropout / 2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))  # Output: predicted return
        
        # Compile with Adam optimizer
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Huber loss is robust to outliers
            metrics=['mae']
        )
        
        return model
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM input.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target array (n_samples,) or None
            
        Returns:
            X_seq: Sequences (n_sequences, lookback, n_features)
            y_seq: Targets for each sequence (n_sequences,) or None
        """
        n_samples = len(X)
        
        if n_samples < self.lookback:
            raise ValueError(
                f"Not enough samples ({n_samples}) for lookback ({self.lookback}). "
                f"Need at least {self.lookback} samples."
            )
        
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(self.lookback, n_samples):
            X_seq.append(X[i - self.lookback:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        if y_seq is not None:
            y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    def fit(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        validation_split: float = 0.1,
    ) -> "LSTMPredictor":
        """
        Fit the LSTM model.
        
        Args:
            X: Feature DataFrame with shape (n_samples, n_features)
            y: Target Series with predicted returns
            validation_split: Fraction of data for validation
            
        Returns:
            self
        """
        if not HAS_TF:
            raise ImportError("TensorFlow not available")
        
        # Store feature names
        self.feature_names = list(X.columns)
        n_features = len(self.feature_names)
        
        # Handle NaN values
        X_clean = X.fillna(0).values
        y_clean = y.fillna(0).values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Scale target (helps with training stability)
        self.target_scaler = StandardScaler()
        y_scaled = self.target_scaler.fit_transform(y_clean.reshape(-1, 1)).ravel()
        
        # Create sequences
        try:
            X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        except ValueError as e:
            logger.warning(f"Cannot create sequences: {e}")
            raise
        
        if len(X_seq) < 10:
            raise ValueError(f"Not enough sequences ({len(X_seq)}) for training")
        
        # Build model
        self.model = self._build_model(n_features)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=self.verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-6,
                verbose=self.verbose
            ),
        ]
        
        # Train
        logger.info(f"Training LSTM with {len(X_seq)} sequences, {n_features} features")
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=self.verbose,
            shuffle=False,  # Keep time order for validation
        )
        
        self.history = history.history
        self.is_fitted = True
        
        # Log training results
        final_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [final_loss])[-1]
        logger.info(f"LSTM training complete. Loss: {final_loss:.6f}, Val Loss: {final_val_loss:.6f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> float:
        """
        Predict return for the next period.
        
        Args:
            X: Recent feature DataFrame with at least `lookback` rows
            
        Returns:
            Predicted return (single float value)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Ensure we have enough data
        if len(X) < self.lookback:
            logger.warning(
                f"Only {len(X)} samples provided, need {self.lookback}. "
                "Padding with zeros."
            )
            # Pad with zeros if needed
            padding = pd.DataFrame(
                np.zeros((self.lookback - len(X), len(self.feature_names))),
                columns=self.feature_names
            )
            X = pd.concat([padding, X], ignore_index=True)
        
        # Use only the columns we trained on
        X_aligned = X[self.feature_names].tail(self.lookback).fillna(0).values
        
        # Scale
        X_scaled = self.scaler.transform(X_aligned)
        
        # Reshape for LSTM: (1, lookback, n_features)
        X_seq = X_scaled.reshape(1, self.lookback, -1)
        
        # Predict
        y_pred_scaled = self.model.predict(X_seq, verbose=0)[0, 0]
        
        # Inverse transform
        y_pred = self.target_scaler.inverse_transform([[y_pred_scaled]])[0, 0]
        
        return float(y_pred)
    
    def predict_proba(self, X: pd.DataFrame) -> float:
        """
        Get probability of positive return.
        
        Uses a sigmoid transformation of the predicted return
        scaled by historical volatility.
        
        Args:
            X: Recent feature DataFrame
            
        Returns:
            Probability of positive return (0 to 1)
        """
        pred_return = self.predict(X)
        
        # Use sigmoid with scaling based on typical return magnitude
        # A 1% predicted return maps to roughly 73% probability
        prob = 1.0 / (1.0 + np.exp(-pred_return * 100))
        
        return float(np.clip(prob, 0.01, 0.99))
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters (sklearn compatibility)."""
        return {
            'lookback': self.lookback,
            'lstm_units': self.lstm_units,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'n_layers': self.n_layers,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }
    
    def set_params(self, **params) -> "LSTMPredictor":
        """Set model parameters (sklearn compatibility)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def summary(self) -> str:
        """Get model summary string."""
        if self.model is None:
            return "Model not built yet"
        
        # Capture model summary
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return '\n'.join(lines)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        import pickle
        
        # Save Keras model
        self.model.save(f"{path}_keras.h5")
        
        # Save scalers and metadata
        metadata = {
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'feature_names': self.feature_names,
            'params': self.get_params(),
            'is_fitted': self.is_fitted,
        }
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> "LSTMPredictor":
        """Load model from disk."""
        import pickle
        
        # Load metadata
        with open(f"{path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(**metadata['params'])
        instance.scaler = metadata['scaler']
        instance.target_scaler = metadata['target_scaler']
        instance.feature_names = metadata['feature_names']
        instance.is_fitted = metadata['is_fitted']
        
        # Load Keras model
        instance.model = keras.models.load_model(f"{path}_keras.h5")
        
        return instance


class LSTMWrapper:
    """
    Sklearn-compatible wrapper for LSTMPredictor.
    
    This allows LSTM to be used with the existing model pipeline
    that expects sklearn-style fit/predict interface.
    """
    
    def __init__(self, **kwargs):
        """Initialize with LSTMPredictor parameters."""
        self.lstm = LSTMPredictor(**kwargs)
        self.feature_names_in_: Optional[List[str]] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "LSTMWrapper":
        """Fit the model."""
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        self.feature_names_in_ = list(X.columns) if hasattr(X, 'columns') else None
        self.lstm.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict returns.
        
        Note: For LSTM, we predict one value at a time using the
        lookback window. For batch predictions, we slide the window.
        """
        if isinstance(X, np.ndarray):
            if self.feature_names_in_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)
        
        # For a single prediction (most common case)
        if len(X) <= self.lstm.lookback:
            pred = self.lstm.predict(X)
            return np.array([pred])
        
        # For batch predictions, slide the window
        predictions = []
        for i in range(self.lstm.lookback, len(X) + 1):
            window = X.iloc[i - self.lstm.lookback:i]
            pred = self.lstm.predict(window)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters."""
        return self.lstm.get_params(deep)
    
    def set_params(self, **params) -> "LSTMWrapper":
        """Set parameters."""
        self.lstm.set_params(**params)
        return self


def create_lstm_model(
    lookback: int = 60,
    lstm_units: int = 50,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 32,
    **kwargs
) -> LSTMWrapper:
    """
    Factory function to create an LSTM model.
    
    Args:
        lookback: Window size for sequences
        lstm_units: Number of LSTM units
        dropout: Dropout rate
        epochs: Training epochs
        batch_size: Batch size
        **kwargs: Additional LSTMPredictor parameters
        
    Returns:
        LSTMWrapper instance
    """
    if not HAS_TF:
        raise ImportError(
            "TensorFlow required for LSTM. Install with: pip install tensorflow"
        )
    
    return LSTMWrapper(
        lookback=lookback,
        lstm_units=lstm_units,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        **kwargs
    )
