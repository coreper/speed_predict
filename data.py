import numpy as np
import pandas as pd
import os
import random
import pickle
from sklearn import preprocessing
import config
args = config.config()


def prep_data(df):
    df_np = df.values

    dist_np = df_np[:, 3]
    speed_np = df_np[:, 4]
    heading_np = df_np[:, 5]
    ax_np = df_np[:, 6]
    ay_np = df_np[:, 7]
    yaw_np = df_np[:, 8]
    radius_np = df_np[:, 9]
    status_np = df_np[:, 10]
    instance = df_np[:, 11]


    dist_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    speed_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    heading_sacler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    ax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    ay_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    yaw_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    radius_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


    dist_scaled = dist_scaler.fit_transform(dist_np.reshape(-1, 1))
    speed_scaled = speed_scaler.fit_transform(speed_np.reshape(-1, 1))
    heading_scaled = heading_sacler.fit_transform(heading_np.reshape(-1, 1))
    ax_scaled = ax_scaler.fit_transform(ax_np.reshape(-1, 1))
    ay_scaled = ay_scaler.fit_transform(ay_np.reshape(-1, 1))
    yaw_scaled = yaw_scaler.fit_transform(yaw_np.reshape(-1, 1))
    radius_scaled = radius_scaler.fit_transform(radius_np.reshape(-1, 1))


    if not os.path.isfile(args.train_data_file) or args.regen == True:
        print('- Regenerate training and testing data')
        data_scale_np = np.column_stack((np.squeeze(dist_scaled),
                                         np.squeeze(speed_scaled),
                                         np.squeeze(heading_scaled),
                                         np.squeeze(ax_scaled),
                                         np.squeeze(ay_scaled),
                                         np.squeeze(yaw_scaled),
                                         np.squeeze(radius_scaled),
                                         np.squeeze(status_np),
                                         instance))
        data_scale_df = pd.DataFrame(data_scale_np, columns=['Dist', 'Speed', 'Heading', 'Ax', 'Ay', 'Yawrate',
                                                             'RadiusOfCurve', 'Status', 'Instance'])

        data_list = [x for _, x in data_scale_df.groupby('Instance')]
        random.shuffle(data_list)
        data_, test_data = data_list[:-300], data_list[-300:]

        all_data_ = []
        trajectory_secs = []
        for instance_df in data_:
            trajectory_sec = instance_df.shape[0]/10
            trajectory_secs.append(trajectory_sec)

            instance_np = instance_df.values
            # Dist, Speed, Heading, Ax, Ay, Yaw rate, RadiusOfCurve, Status, Instance
            x_np = instance_np[:, :args.x_dim]
            speed = instance_np[:, 1]

            for i in range(int(len(instance_np)/30)):
                try:
                    rand_position = random.randint(args.ref_time*10, len(x_np) - args.pred_time*10)
                except ValueError:
                    continue
                x = x_np[rand_position - args.ref_time*10:rand_position, :]
                y = speed[rand_position:rand_position + args.pred_time*10]
                y = y[::int(args.pred_time*10 / args.points)]
                all_data_.append([x, y])


        train_data = all_data_[:int(args.train_ratio * len(all_data_))]
        val_data = all_data_[int(args.train_ratio * len(all_data_)):]

        # Save the preprocessed data to file
        with open(args.train_data_file, 'wb') as fp:
            pickle.dump(train_data, fp)
        with open(args.val_data_file, 'wb') as fp:
            pickle.dump(val_data, fp)
        with open(args.test_data_file, 'wb') as fp:
            pickle.dump(test_data, fp)

    else:
        print('- Reading all training and testing data')
        with open(args.train_data_file, 'rb') as fp:
            train_data = pickle.load(fp)
        with open(args.val_data_file, 'rb') as fp:
            val_data = pickle.load(fp)
        with open(args.test_data_file, 'rb') as fp:
            test_data = pickle.load(fp)

    return train_data, val_data, test_data, speed_scaler, dist_scaler