"""
PyTorch Neural Network Training Example
========================================

Demonstrates neural network training for biomechanical prediction:
1. Prepare training data from multiple tests
2. Build custom neural network architecture
3. Train model with TorchTrainer
4. Implement early stopping and learning rate scheduling
5. Evaluate model performance
6. Visualize training history and predictions

Common use case: 1RM prediction, injury risk modeling, performance prediction.
"""

import labanalysis as laban
from labanalysis.modelling.pytorch import (
    TorchTrainer,
    FeaturesGenerator,
    CustomDataset
)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # ===== 1. PREPARE TRAINING DATA =====
    print("Preparing training data...")

    # Example: Predict 1RM from isokinetic test metrics
    # In practice, load from multiple athlete files

    # Simulated training data (replace with actual data loading)
    np.random.seed(42)
    n_samples = 200

    # Features: peak force, peak velocity, peak power, time to peak, work
    features = {
        'peak_force': np.random.uniform(800, 2000, n_samples),
        'peak_velocity': np.random.uniform(0.5, 2.5, n_samples),
        'peak_power': np.random.uniform(400, 1200, n_samples),
        'time_to_peak': np.random.uniform(0.1, 0.5, n_samples),
        'total_work': np.random.uniform(200, 600, n_samples)
    }

    # Target: 1RM (kg)
    # Simplified relationship for demonstration
    one_rm = (
        0.05 * features['peak_force'] +
        100 * features['peak_velocity'] +
        0.15 * features['peak_power'] +
        np.random.normal(0, 20, n_samples)  # Add noise
    )

    # Create DataFrame
    df = pd.DataFrame(features)
    df['one_rm'] = one_rm

    print(f"Dataset: {n_samples} samples")
    print(f"Features: {list(features.keys())}")
    print(f"Target range: {one_rm.min():.1f} - {one_rm.max():.1f} kg")


    # ===== 2. DATA PREPROCESSING =====
    print("\n===== DATA PREPROCESSING =====")

    # Split into train/val/test sets
    X = df[list(features.keys())].values
    y = df['one_rm'].values.reshape(-1, 1)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=42  # 0.18 * 0.85 ≈ 0.15
    )

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Feature scaling (important for neural networks)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)

    print("\n✓ Features standardized (mean=0, std=1)")


    # ===== 3. CREATE PYTORCH DATASETS =====
    print("\nCreating PyTorch datasets...")

    train_dataset = CustomDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train_scaled)
    )

    val_dataset = CustomDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val_scaled)
    )

    test_dataset = CustomDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test_scaled)
    )


    # ===== 4. DEFINE NEURAL NETWORK =====
    print("\n===== NEURAL NETWORK ARCHITECTURE =====")

    class OneRMPredictor(nn.Module):
        """Neural network for 1RM prediction."""

        def __init__(self, input_dim, hidden_dims=[64, 32, 16], dropout=0.2):
            super().__init__()

            layers = []
            prev_dim = input_dim

            # Hidden layers
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Create model
    input_dim = X_train.shape[1]
    model = OneRMPredictor(
        input_dim=input_dim,
        hidden_dims=[64, 32, 16],
        dropout=0.2
    )

    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")


    # ===== 5. CONFIGURE TRAINING =====
    print("\n===== TRAINING CONFIGURATION =====")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    print("Loss: MSE")
    print("Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
    print("Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")


    # ===== 6. TRAIN MODEL =====
    print("\n===== TRAINING =====")

    # Create TorchTrainer
    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Device: {trainer.device}")

    # Train with early stopping
    history = trainer.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=200,
        batch_size=32,
        early_stopping_patience=20,
        verbose=True
    )

    print(f"\n✓ Training completed")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")


    # ===== 7. EVALUATE MODEL =====
    print("\n===== EVALUATION =====")

    # Predict on test set
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(trainer.device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Calculate metrics
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest Set Performance:")
    print(f"  MAE:  {mae:.2f} kg")
    print(f"  RMSE: {rmse:.2f} kg")
    print(f"  R²:   {r2:.4f}")

    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"  MAPE: {mape:.2f}%")


    # ===== 8. CREATE VISUALIZATIONS =====
    print("\nCreating visualizations...")

    # Plot 1: Training history
    fig1 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Loss Curves', 'Learning Rate')
    )

    # Loss curves
    fig1.add_trace(
        go.Scatter(
            x=list(range(1, len(history['train_loss']) + 1)),
            y=history['train_loss'],
            mode='lines',
            name='Train Loss',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig1.add_trace(
        go.Scatter(
            x=list(range(1, len(history['val_loss']) + 1)),
            y=history['val_loss'],
            mode='lines',
            name='Val Loss',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # Mark best epoch
    fig1.add_vline(
        x=history['best_epoch'],
        line_dash="dash",
        line_color="green",
        annotation_text="Best",
        row=1, col=1
    )

    # Learning rate
    if 'learning_rates' in history:
        fig1.add_trace(
            go.Scatter(
                x=list(range(1, len(history['learning_rates']) + 1)),
                y=history['learning_rates'],
                mode='lines',
                name='LR',
                line=dict(color='purple', width=2)
            ),
            row=1, col=2
        )

    fig1.update_xaxes(title_text="Epoch", row=1, col=1)
    fig1.update_xaxes(title_text="Epoch", row=1, col=2)
    fig1.update_yaxes(title_text="Loss", row=1, col=1)
    fig1.update_yaxes(title_text="Learning Rate", type="log", row=1, col=2)

    fig1.update_layout(
        title="Training History",
        height=400,
        template='plotly_white'
    )


    # Plot 2: Predictions vs Actual
    fig2 = go.Figure()

    # Scatter plot
    fig2.add_trace(
        go.Scatter(
            x=y_test.flatten(),
            y=y_pred.flatten(),
            mode='markers',
            marker=dict(size=8, color='blue', opacity=0.6),
            name='Predictions'
        )
    )

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    fig2.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect prediction'
        )
    )

    fig2.update_layout(
        title=f"Predictions vs Actual (R² = {r2:.3f}, MAE = {mae:.1f} kg)",
        xaxis_title="Actual 1RM (kg)",
        yaxis_title="Predicted 1RM (kg)",
        template='plotly_white'
    )


    # Plot 3: Prediction errors
    errors = y_test.flatten() - y_pred.flatten()

    fig3 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Error Distribution', 'Error vs Actual')
    )

    # Error histogram
    fig3.add_trace(
        go.Histogram(
            x=errors,
            nbinsx=30,
            marker_color='blue',
            opacity=0.7,
            name='Errors'
        ),
        row=1, col=1
    )

    # Error vs actual
    fig3.add_trace(
        go.Scatter(
            x=y_test.flatten(),
            y=errors,
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.6),
            name='Errors'
        ),
        row=1, col=2
    )

    fig3.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

    fig3.update_xaxes(title_text="Error (kg)", row=1, col=1)
    fig3.update_xaxes(title_text="Actual 1RM (kg)", row=1, col=2)
    fig3.update_yaxes(title_text="Count", row=1, col=1)
    fig3.update_yaxes(title_text="Error (kg)", row=1, col=2)

    fig3.update_layout(
        title="Prediction Errors",
        height=400,
        template='plotly_white',
        showlegend=False
    )


    # ===== 9. DISPLAY PLOTS =====
    print("\nDisplaying plots...")
    fig1.show()
    fig2.show()
    fig3.show()


    # ===== 10. SAVE MODEL =====
    print("\nSaving model...")

    # Save PyTorch model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_names': list(features.keys()),
        'architecture': {
            'input_dim': input_dim,
            'hidden_dims': [64, 32, 16],
            'dropout': 0.2
        }
    }, "one_rm_predictor.pth")

    print("✓ Saved: one_rm_predictor.pth")


    # ===== 11. EXPORT RESULTS =====
    print("\nExporting results...")

    # Predictions table
    results_df = pd.DataFrame({
        'Actual_1RM': y_test.flatten(),
        'Predicted_1RM': y_pred.flatten(),
        'Error': errors,
        'Abs_Error': np.abs(errors),
        'Percent_Error': np.abs(errors) / y_test.flatten() * 100
    })

    results_df.to_csv("predictions_results.csv", index=False)
    print("✓ Saved: predictions_results.csv")

    # Summary statistics
    summary = {
        'Metric': ['MAE', 'RMSE', 'R²', 'MAPE', 'Best Epoch', 'Best Val Loss'],
        'Value': [
            f"{mae:.2f}",
            f"{rmse:.2f}",
            f"{r2:.4f}",
            f"{mape:.2f}",
            history['best_epoch'],
            f"{history['best_val_loss']:.4f}"
        ],
        'Unit': ['kg', 'kg', '-', '%', '-', '-']
    }

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("training_summary.csv", index=False)
    print("✓ Saved: training_summary.csv")


    print("\n===== TRAINING COMPLETE =====")
    print(f"Model performance: MAE = {mae:.1f} kg, R² = {r2:.3f}")
    print(f"Model saved to: one_rm_predictor.pth")


if __name__ == "__main__":
    main()
