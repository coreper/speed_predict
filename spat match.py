import pandas as pd
import os
import numpy as np

data_folder = './MatchedTripsAndSpat_2'
trip_folder = './Trips'
if not os.path.exists(trip_folder):
    os.makedirs(trip_folder)

for file in os.listdir(data_folder):
    if file.endswith('-BSM.csv'):
        bsm_file = os.path.join(data_folder, file)
        spat_file = os.path.join(data_folder, file[:-7]+'spat.csv')
        print('working on', bsm_file)

        bsm = pd.read_csv(bsm_file)
        spat = pd.read_csv(spat_file)

        bsm_head = bsm.columns
        c_name = list(bsm_head)
        c_name.pop()
        c_name.append('Status')

        for trip in bsm.groupby('Trip ID'):
            trajectory = trip[1]
            trip_id = str(trajectory['Trip ID'].unique()[0])

            output = pd.DataFrame(columns=c_name)
            for i in range(len(trajectory)):
                point = trajectory.iloc[i]
                time = point['GpsEpochTime']
                phase = point[' Phase No']
                # trajectory not passing the intersection
                if phase==-1:
                    continue
                else:
                    values = point.tolist()
                    values.pop()
                    phaseid= 'Phase'+str(phase)
                    subject = spat.loc[
                        (spat[' Start_time'] <= time) & (spat[' End_time'] >= time) & (spat[' Phase_ID'] == phaseid)]
                    status = subject[' Phase_Status'].tolist()
                    if status != []:
                        s = status[0]
                        values.append(s)
                        a_series = pd.Series(values, index = output.columns)

                        output = output.append(a_series, ignore_index=True)
                    else:
                        print(trip_id, phaseid, point['Gentime'], time)
                        continue
            if output.empty:
                print(trip_id, 'Empty')
            else:
                output.to_csv(os.path.join(trip_folder, file[:-7]+trip_id+'.csv'), index=False)
