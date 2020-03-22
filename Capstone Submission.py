#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[3]:


#Scraping Canada Postal Codes from Wiki into a dataframe
url='https://en.wikipedia.org/w/index.php?title=List_of_postal_codes_of_Canada:_M&oldid=890001695.'
df=pd.read_html(url, header=0)[0]
df.rename(columns={'Postcode':'PostalCode'},inplace = True)
print(df.shape)
df.head()


# In[4]:


#Deleting rows with Borough as Not assigned
df.drop(df[df.Borough == 'Not assigned'].index, inplace = True)
print(df.shape)
df.head()


# In[5]:


#Clubing rows with same postcodes
aggregation_functions = {'Borough': 'first', 'Neighbourhood':lambda col: ','.join(col)}
df = df.groupby(df['PostalCode']).aggregate(aggregation_functions)
print(df.shape)
df.head()


# In[6]:


#Replace not assigned value of Neighbourhood from Borough
df.loc[df['Neighbourhood'] == 'Not assigned']
df.loc[df['Neighbourhood'] == 'Not assigned'] = df['Borough']
df.loc[df['Neighbourhood'] == 'Not assigned']
print(df.shape)
df.head()


# In[8]:


#Loading latitude and longitude csv file
df_geoco=pd.read_csv("https://cocl.us/Geospatial_data")


# In[9]:


df_geoco.rename(columns = {'Postal Code':'PostalCode'}, inplace=True)
print(df_geoco.shape)
df_geoco.head()


# In[10]:


#Merging coordinates with postal codes dataframe
df_merged = pd.merge(df, df_geoco, how='left', on=['PostalCode'])
print(df_merged.shape)
df_merged.head()


# In[11]:


#Limiting our exploration to borough's containing word 'Toronto'
#Filtering DF by boroughs containing word 'Toronto'

df_merged = df_merged[df_merged['Borough'].str.contains('Toronto')]
print(df_merged.shape)
df_merged.head()


# In[12]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(df_merged['Borough'].unique()),
        df_merged.shape[0]
    )
)


# In[13]:


conda install -c conda-forge geopy


# In[15]:


#Getting coordinates of Toronto
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import folium
import requests
from pandas.io.json import json_normalize

address = 'Toronto, Canada'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude, longitude))


# In[16]:


#Getting folium map highlighting each neighborhood using markers

map_toronto = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, borough, neighborhood in zip(df_merged['Latitude'], df_merged['Longitude'], df_merged['Borough'], df_merged['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='orange',
        fill=True,
        fill_color='yellow',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto


# In[17]:


#Exloring the first borough - East Toronto
#Getting coordinates of East Toronto

df_eastToronto = df_merged[df_merged['Borough'] == 'East Toronto'].reset_index(drop=True)
df_eastToronto.head()


# In[18]:


address = 'East Toronto, Toronto'

geolocator = Nominatim(user_agent="toronto_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of East Toronto are {}, {}.'.format(latitude, longitude))


# In[19]:


#Plotting map for East Toronto highlighting all neighbourhoods in the borough

map_eastToronto = folium.Map(location=[latitude, longitude], zoom_start=10)

for lat, lng, borough, neighborhood in zip(df_eastToronto['Latitude'], df_eastToronto['Longitude'], df_eastToronto['Borough'], df_eastToronto['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='orange',
        fill=True,
        fill_color='yellow',
        fill_opacity=0.7,
        parse_html=False).add_to(map_eastToronto)  
    
map_eastToronto


# In[20]:


#Exploring the first neighborhood in East Toronto - 'The Beaches'

df_eastToronto.loc[0, 'Neighbourhood']


# In[21]:


neigh_latitude = df_eastToronto.loc[0, 'Latitude']
neigh_longitude = df_eastToronto.loc[0, 'Longitude']
neigh_name = df_eastToronto.loc[0, 'Neighbourhood']

print('Latitude and Longitude of {} are {}, {}'.format(neigh_name, neigh_latitude, neigh_longitude))


# In[22]:


#Getting venues around the above neighbourhood
#Foursquare Credentials

CLIENT_ID = 'FSZLRDVS0SYVW1T3NJN32EUXOBKSRVCIZLFPD3M1QCDVRZBA'
CLIENT_SECRET = 'TV3FU1RMYS5WBVZNDXQZYVTE4ZV5UC22KBK4HNF4TPT4G2SD'
VERSION = '20180605'

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[23]:


LIMIT = 100
radius = 500
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neigh_latitude, 
    neigh_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[24]:


results = requests.get(url).json()


# In[25]:


#Defining a function that extracts the category of the venue

def get_category_type(row):
    try:
        categories_list = row['categoriaes']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[27]:


#Extracting venue info from the API call

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
nearby_venues.head()


# In[28]:


#Repeating the process for all neighbourhoods in East Toronto
#Defining a function that returns nearby venues for a list of places

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[29]:


#Calling the above function for our list of neighborhoods

eastToronto_venues = getNearbyVenues(names=df_eastToronto['Neighbourhood'],
                                   latitudes=df_eastToronto['Latitude'],
                                   longitudes=df_eastToronto['Longitude']
                                  )


# In[30]:


#List of venues for each of our neighborhoods

print(eastToronto_venues.shape)
eastToronto_venues.head()


# In[31]:


#A count of venues per neighborhood

eastToronto_venues.groupby('Neighborhood').count()


# In[32]:


#Analyzing each neighborhood

# one hot encoding
eastToronto_onehot = pd.get_dummies(eastToronto_venues[['Venue Category']], prefix="", prefix_sep="")

eastToronto_onehot.drop(labels=['Neighborhood'], axis=1,inplace = True)

eastToronto_onehot['Neighborhood'] = eastToronto_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [eastToronto_onehot.columns[-1]] + list(eastToronto_onehot.columns[:-1])
eastToronto_onehot = eastToronto_onehot[fixed_columns]

eastToronto_onehot.head()


# In[33]:


#Frequency of each venue

eastToronto_grouped = eastToronto_onehot.groupby('Neighborhood').mean().reset_index()
eastToronto_grouped


# In[34]:


num_top_venues = 5

for hood in eastToronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = eastToronto_grouped[eastToronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[35]:


#A list of most common venues for each neighborhood

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[36]:



indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = eastToronto_grouped['Neighborhood']

for ind in np.arange(eastToronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(eastToronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[37]:


#Clustering neighborhoods using k-means

# set number of clusters
kclusters = 5

eastToronto_grouped_clustering = eastToronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(eastToronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]


# In[38]:


#neighborhoods_venues_sorted.drop('Cluster Labels')
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)


# In[39]:


#Merging our original table with the common venues

eastToronto_merged = df_eastToronto

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
eastToronto_merged = eastToronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighbourhood')

eastToronto_merged.head() # Last columns


# In[40]:


eastToronto_merged['Cluster Labels']


# In[41]:


#Plotting the clusters on map

map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(eastToronto_merged['Latitude'], eastToronto_merged['Longitude'], eastToronto_merged['Neighbourhood'], eastToronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# In[43]:


#Examining CLusters

eastToronto_merged.loc[eastToronto_merged['Cluster Labels'] == 0, eastToronto_merged.columns[[1] + list(range(5, eastToronto_merged.shape[1]))]]


# In[44]:


eastToronto_merged.loc[eastToronto_merged['Cluster Labels'] == 1, eastToronto_merged.columns[[1] + list(range(5, eastToronto_merged.shape[1]))]]


# In[45]:


eastToronto_merged.loc[eastToronto_merged['Cluster Labels'] == 2, eastToronto_merged.columns[[1] + list(range(5, eastToronto_merged.shape[1]))]]


# In[46]:


eastToronto_merged.loc[eastToronto_merged['Cluster Labels'] == 4, eastToronto_merged.columns[[1] + list(range(5, eastToronto_merged.shape[1]))]]


# In[ ]:




