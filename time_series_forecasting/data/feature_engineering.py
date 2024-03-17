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

    def christmas_eve_feature(self, df):
        df['christmas_eve'] = ((df['timestamp'].dt.month == 12) & (df['timestamp'].dt.day == 24)).astype(int)
        return df

    def add_features(self, df):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = self.hour_feature(df)
        df = self.weekday_feature(df)
        df = self.week_feature(df)
        df = self.christmas_eve_feature(df)
        return df

    def select_features(self, df, features: List[str]):
        return df[features]
