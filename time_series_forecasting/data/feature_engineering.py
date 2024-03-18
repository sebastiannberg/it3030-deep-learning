import numpy as np
import pandas as pd
from typing import List


class FeatureEngineering:

    def hour_feature(self, df):
        df['hour'] = df['timestamp'].dt.hour
        # Apply sine and cosine transformations
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        return df

    def weekday_feature(self, df):
        df['weekday'] = df['timestamp'].dt.weekday
        # Apply sine and cosine transformations
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7.0)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7.0)
        return df

    def week_feature(self, df):
        df['week'] = df['timestamp'].dt.isocalendar().week
        # Apply sine and cosine transformations
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 53.0)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 53.0)
        return df

    def month_feature(self, df):
        df['month'] = df['timestamp'].dt.month
        # Apply sine and cosine transformations
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        return df

    def lag_features(self, df, bidding_area):
        # Consumption 48 hours ago
        df["lag_48h_consumption"] = df[f"{bidding_area}_consumption"].shift(48)

        # Mean consumption of the previous day
        df['lag_mean_24h_consumption'] = df[f"{bidding_area}_consumption"].rolling(window=24).mean().shift(24)

        return df

    def add_features(self, df, bidding_area):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = self.hour_feature(df)
        df = self.weekday_feature(df)
        df = self.week_feature(df)
        df = self.month_feature(df)
        df = self.lag_features(df, bidding_area)
        # Drop rows with NaN after introducing lag features
        df.dropna(inplace=True)
        return df
