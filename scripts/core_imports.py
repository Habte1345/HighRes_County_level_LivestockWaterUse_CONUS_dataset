# core_imports.py

import sys
import os
import re
import io
import zipfile
import warnings
from typing import Dict, Any
import subprocess
# Data Handling & Scientific Libraries
import numpy as np
import pandas as pd
import requests
from scipy.interpolate import UnivariateSpline, PchipInterpolator
import statsmodels.api as sm

# Plotting & Geospatial
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import geopandas as gpd
import proplot as pplt
import ee
import pyarrow

# Scikit-learn (ML) Libraries
from sklearn.model_selection import (
    train_test_split, GridSearchCV, KFold, cross_val_predict, 
    cross_validate, GroupKFold, cross_val_score
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# Tensorflow/Keras (DL) Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # For inset plot
import proplot as pplt # Assuming pplt is proplot
import geopandas as gpd # Assuming CONUS_counties and other dataframes are GeoDataFrames
import numpy as np
