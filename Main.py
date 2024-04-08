import pandas as pd
import numpy as np
import os.path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('New_Police_Incidents.csv')




data.replace(to_replace="BEHIND THE ROCKS", value=str(0), inplace=True)

data.replace(to_replace="UPPER ALBANY", value=str(1), inplace=True)
data.replace(to_replace="BARRY SQUARE", value=str(2), inplace=True)
data.replace(to_replace="DOWNTOWN", value=str(3), inplace=True)
data.replace(to_replace="BLUE HILLS", value=str(4), inplace=True)
data.replace(to_replace="SHELDON-CHARTER OAK", value=str(5), inplace=True)
data.replace(to_replace="NORTHEAST", value=str(6), inplace=True)
data.replace(to_replace="SOUTH GREEN", value=str(7), inplace=True)
data.replace(to_replace="SOUTHEND", value=str(8), inplace=True)
data.replace(to_replace="FROG HOLLOW", value=str(9), inplace=True)
data.replace(to_replace="WESTEND", value=str(10), inplace=True)
data.replace(to_replace="PARKVILLE", value=str(11), inplace=True)
data.replace(to_replace="CLAY-ARSENAL", value=str(12), inplace=True)
data.replace(to_replace="ASYLUM HILL", value=str(13), inplace=True)
data.replace(to_replace="NORTH MEADOWS", value=str(14), inplace=True)
data.replace(to_replace="SOUTHWEST", value=str(15), inplace=True)
data.replace(to_replace="SOUTH MEADOWS", value=str(16), inplace=True)




if os.path.exists('New_Police_Incidents_V1.csv'):
     print('file exists')
else:
     data.to_csv('New_Police_IncidentsV1.csv')