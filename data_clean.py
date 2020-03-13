from geopy.distance import great_circle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import utilities as ut
import gmplot as gplt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import glob
import os

# extract data and match with spat below
cwd = os.getcwd()
# print(cwd)
BSM_files = glob.glob("./MatchedTripsAndSpat/*-BSM.csv")
spat_files = glob.glob("./MatchedTripsAndSpat/*-spat.csv")
# print(trajectory_files,"\n")
# print(spat_files)

# get all files' unique name, which is the date
unique_id = []
for element in BSM_files:
    start = element.find('2019')
    end = element.find('_62606176')
    unique_id.append(int(element[start:end]))
unique_id = np.sort(unique_id)

for date in unique_id:
    BSM_filepath = "./MatchedTripsAndSpat/" + str(date) + '_62606176' + "-BSM.csv"
    spat_filepath = "./MatchedTripsAndSpat/" + str(date) + '_62606176' + "-spat.csv"
    print("matching %s and %s" % (BSM_filepath, spat_filepath))
    print("loading corresponding BSM and spat file")
    bsm = pd.read_csv(BSM_filepath)
    spat = pd.read_csv(spat_filepath)
    bsm_head = bsm.columns
    print("loaded")

    # matching algo below
    tripID = set(bsm['Trip ID'])
    c_name = list(bsm_head)
    c_name.pop()
    c_name.append('Status')
    output = pd.DataFrame(columns=c_name)  # use output to store the results and save to file

    for tripid in tripID:
        print("now examing" + str(tripid))
        trajectory = pd.DataFrame(bsm.loc[bsm['Trip ID'] == tripid])
        for i in range(len(trajectory)):
            point = trajectory.iloc[i]
            time = point['GpsEpochTime']
            phase = point[' Phase No']
            # trajectory not passing the intersection
            if phase == -1:
                continue
            else:
                values = point.tolist()
                values.pop()
                phaseid = 'Phase' + str(phase)
                subject = spat.loc[
                    (spat[' Start_time'] <= time) & (spat[' End_time'] >= time) & (spat[' Phase_ID'] == phaseid)]
                status = subject[' Phase_Status'].tolist()
                if status != []:
                    #             print(status)
                    s = status[0]
                    values.append(s)
                    #             print(values)
                    a_series = pd.Series(values, index=output.columns)

                    output = output.append(a_series, ignore_index=True)
                else:
                    #                     print(phaseid, point['Gentime'], time)
                    continue
    print("BSM and spat matched, saving file")
    output_path = cwd + "/matched/matched_" + str(date) + ".csv"
    output.to_csv(output_path)
    print("matched file was saved to:" + output_path)
