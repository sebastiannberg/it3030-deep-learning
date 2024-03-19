import numpy as np
import torch


class Preprocessor:

    def __init__(self):
        self.spike_params = {}
        self.scaler_params = {}

    def get_params(self):
        return self.spike_params, self.scaler_params

    def remove_spikes(self, df, features, fit=False):
        """
        Detect and remove spikes using a rolling window median.
        """
        df_processed = df.copy()

        if fit:
            for col in features:
                moving_median = df_processed[col].rolling(window=12, center=True).median()
                diff_from_median = (df_processed[col] - moving_median).abs()
                threshold = diff_from_median.quantile(0.9995)
                self.spike_params[col] = threshold
        else:
            # Use previously stored thresholds if not fitting
            assert self.spike_params, "Spike removal parameters not initialized. Call with fit=True first."

        total_spikes_detected = 0
        for col in features:
            threshold = self.spike_params.get(col)
            moving_median = df_processed[col].rolling(window=12, center=True).median()
            diff_from_median = (df_processed[col] - moving_median).abs()
            spike_mask = diff_from_median > threshold
            total_spikes_detected += spike_mask.sum()
            df_processed.loc[spike_mask, col] = np.nan

        print(f"Detected {total_spikes_detected} spikes using rolling window median")
        # Fill holes created by spike removal wiht interpolation
        # Interpolate only the specified features
        for col in features:
            df_processed[col].interpolate(method='polynomial', order=2, inplace=True, limit_direction='both')

        return df_processed

    def standardize(self, df, features, fit=False):
        print(f"Standardizing data (fit={fit})")
        standardized_df = df.copy()
        if fit:
            # Calculate mean and std
            self.scaler_params = {col: {'mean': df[col].mean(), 'std': df[col].std()} for col in features}

        # Apply standardization only to specified features
        for col in features:
            if col in self.scaler_params:
                mean = self.scaler_params[col]['mean']
                std = self.scaler_params[col]['std']
                standardized_df[col] = (df[col] - mean) / std

        return standardized_df

    def reverse_standardize_targets(self, tensor: torch.Tensor):
        # Tensor should be size (minibatch_size, forecast_horizon) eg. (32, 24)
        # Or else it should be (forecast_horizon) eg. (24)
        # Important: it should only be used for target feature eg. consumption
        if not self.scaler_params:
            raise ValueError("Scaler parameters not initialized. Call standardize with fit=True first.")

        # TODO remove hard coded value
        params = self.scaler_params["NO1_consumption"]
        mean = params['mean']
        std = params['std']
        # Reverse the standardization
        tensor = tensor * std + mean

        return tensor
