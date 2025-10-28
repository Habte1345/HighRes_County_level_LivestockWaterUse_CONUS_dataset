# core_imports.py

import sys
import os
import re
import io
import zipfile
import warnings
from typing import Dict, Any

# Data Handling & Scientific Libraries
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import statsmodels.api as sm

# Scikit-learn (ML) Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict, cross_validate, GroupKFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline

# Tensorflow/Keras (DL) Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Geospatial/External Libraries
import ee
# Note: You still need to ensure pyarrow is installed in your environment if you use Feather
import pyarrow