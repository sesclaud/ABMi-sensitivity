import pandas as pd
from geopy import distance
from tqdm import tqdm
import pickle

amenities = pd.read_csv("amenities.csv")

file = open('coords.pkl', 'rb')
coords = pickle.load(file)
file.close()


# identify the closest amenities and build the df
am_counts = pd.DataFrame({i:{'leisure_200': 0, 'food_200': 0, 'school_200': 0, 'office_200': 0, \
                             'leisure_500': 0, 'food_500': 0, 'school_500': 0, 'office_500': 0} for i in coords}).T
for i in tqdm(coords):    
    for j in range(amenities.shape[0]):
        category = amenities.iloc[j]['category']
        if coords[i][1] - amenities.iloc[j]['lat'] > 0.005:
            pass
        elif distance.distance(coords[i], (amenities.iloc[j]['lon'], amenities.iloc[j]['lat'])).km <= 0.2:
            cat1 = category + '_200'
            cat2 = category + '_500'
            am_counts.loc[i][cat1] += 1
            am_counts.loc[i][cat2] += 1
        elif 0.02 < distance.distance(coords[i], (amenities.iloc[j]['lon'], amenities.iloc[j]['lat'])).km <= 0.5:
            cat = category + '_500'
            am_counts.loc[i][cat] += 1

am_counts.to_csv('am_counts_200_500.csv')