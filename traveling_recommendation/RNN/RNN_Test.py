import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt


train_data=pd.read_csv("Data/여행지 테마별 분류_최종본(21.09.06) (2).csv")

print(train_data.head())