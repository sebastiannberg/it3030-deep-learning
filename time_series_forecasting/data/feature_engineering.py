import numpy as np
import pandas as pd


class FeatureEngineering:

    def __init__(self):
        self.added_features = []

    def get_added_features(self):
        return self.added_features

    def hour_feature(self, df):
        df['hour'] = df['timestamp'].dt.hour
        # Apply sine and cosine transformations
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        return df, ['hour', 'hour_sin', 'hour_cos']

    def weekday_feature(self, df):
        df['weekday'] = df['timestamp'].dt.weekday
        # Apply sine and cosine transformations
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)
        return df, ['weekday', 'weekday_sin', 'weekday_cos']

    def week_feature(self, df):
        df['week'] = df['timestamp'].dt.isocalendar().week
        # Apply sine and cosine transformations
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 53.0)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 53.0)
        return df, ['week', 'week_sin', 'week_cos']

    def month_feature(self, df):
        df['month'] = df['timestamp'].dt.month
        # Apply sine and cosine transformations
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        return df, ['month', 'month_sin', 'month_cos']

    def lag_features(self, df, bidding_area):
        # Consumption 48 hours ago
        df["lag_48h_consumption"] = df[f"{bidding_area}_consumption"].shift(48)
        # Mean consumption of the previous day
        df['lag_mean_24h_consumption'] = df[f"{bidding_area}_consumption"].rolling(window=24).mean().shift(24)
        return df, ['lag_48h_consumption', 'lag_mean_24h_consumption']

    def add_features(self, df, bidding_area):
        self.added_features = []
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        for feature_method in [self.hour_feature, self.weekday_feature, self.week_feature, self.month_feature, lambda df: self.lag_features(df, bidding_area)]:
            df, features = feature_method(df)
            self.added_features.extend(features)

        # Drop rows with NaN after introducing lag features
        df.dropna(inplace=True)

        return df
