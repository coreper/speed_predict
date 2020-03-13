import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import geopandas as gpd
import gmplot as gplt
from pyproj import Proj, transform
from geopy.distance import great_circle


# filename = "20190816_62606176-BSM.csv"

class DataFrameProcess():
	def __init__(self):
		self.df = []
		self.vehicleset=[]
		self.positiveset=[]
		
	def readFile(self, filename):
		self.df = pd.read_csv(filename)
		print("get dataframe from %s" %filename)
		
	def getVehicleID(self):
		"""
		return all vehicle ID
		"""
		vehicleSet= (self.df.loc[:,'TxDevice'].values)
		self.vehicleset=list(set(vehicleSet))
		return self.vehicleset
#         print("vehicle set length:", len(vehicleSet),'\n',vehicleSet)

	def getPositiveVehicleID(self):
		vID = self.getVehicleID()
		self.positiveset = [i for i in vID if i>0]
		return self.positiveset
	
	def getOneVehicleTrips(self, vID=8197):
		"""
		return all raw trips of an vehicleID and all trip ID
		"""
		trips = self.df.loc[self.df.loc[:,'TxDevice']==vID,:] # all trip bsm of the vehicle
		tripSet = set(trips.loc[:,'TxRandom'].values) # get the number of trips 
		# print("#get %d total trips from %d" %(len(tripSet), vID))
		return trips, tripSet
	
	def getSpecifiedTrip(self, vID=8197, tripID=-20430):
		trips, tripSet = self.getOneVehicleTrips(vID)
		trip = trips.loc[trips.loc[:,'TxRandom']==tripID,:]
		# print("#get trip %d from %d" %(tripID, vID))
		return trip
	
	def plotTrips(self, dataTrip, tripSet, tagID=000000, basemap=False, axlim=True ):
		"""
		plot all trips in longitude and latitude 
		tagID is used for generate plot title

		"""

		#         dataTrip, tripSet = self.getOneVehicleTrips(vID)

		for i in list(tripSet):
			trip = dataTrip.loc[dataTrip.loc[:,'TxRandom']==i,:]

			gmap = gplt.GoogleMapPlotter(42.30253,-83.70419, 18)
			gmap.apikey = "AIzaSyBJwPcLYZKT_-vCdoYIwkqzuY5PRV25uEs"
			if basemap==True:

				gmap.heatmap(trip['Latitude'],trip['Longitude'])
			elif basemap==False:
				plt.scatter(x=trip['Longitude'], y=trip['Latitude'],s=2)
		if basemap ==True:

			gmap.draw("map.html")
		elif basemap==False:
			if axlim:
				plt.ylim([42.3042,42.3060])
				plt.xlim([-83.6943,-83.6914])
			plt.axis('off')
			plt.tight_layout()
			plt.title("trips at intersection Green+Plymouth")
			plt.show()
		
	# def wgs84_proj(points):
	# 	"""
	# 	return a cartesian numpy array
	# 	"""

	# 	# notes for gmaps datum
#	 	https://gis.stackexchange.com/questions/299396/what-datum-is-used-by-google-maps


	# 	inProj = Proj(init='epsg:4326')
	# 	outProj = Proj(init='epsg:3857')
	# 	result = []
	# 	for point in points:
	# 		# print("point",point)
	# 		lat,longt = transform(inProj,outProj,point[0],point[1])
	# 		# print("lat,long:",lat,longt)
	# 		result.append([lat,longt])
	# 	return np.array(result)

	def getTripOfInterest(self, trips, tripset, intersection=(42.305018, -83.692888)):
		"""
		if all points in a trip is not passing the intersection(20m range),
		the trip is deleted from the tripset 

		return a set that only contains tripID for the given trips that passing the intersection 

		trips are from the same vehicle ID
		"""
		candidates = tripset
		for i in list(candidates):
			trip = trips.loc[trips.loc[:,'TxRandom']==i,:]
			flag =False # default not deleting
			sequence =[x for x in zip(trip['Latitude'],trip['Longitude'])]
			if len(sequence) <100: # less than 100 points
				tripset.remove(i)
			else:
				Distance = []
				
				north_refer=(42.305612, -83.692542)
				west_refer=(42.304248, -83.693948)
				Distance_north=[]
				Distance_west=[]

				# enumerte all points get the distance
				for point in sequence:
					point = tuple(point)
					distance = great_circle(point, intersection).meters
					distance_w = great_circle(point, west_refer).meters
					distance_n = great_circle(point, north_refer).meters
					Distance.append(distance)
					Distance_west.append(distance_w)
					Distance_north.append(distance_n)
				if min(Distance) >20 or min(distance_n)<31 or min(distance_w)<36:
					# print("remove", i)
					flag=True
					tripset.remove(i)
		
				Distance=[]
				Distance_north=[]
				Distance_west=[]
		return tripset
				


		
		


		
