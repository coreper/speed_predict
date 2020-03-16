import numpy as np
import pandas as pd
import os
import random
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import config
args = config.config()

def scale_data(df):
    df_others = df[['Heading', 'Ax', 'Ay', 'Yawrate', 'RadiusOfCurve']]
    df_speed = df[['Speed']]
    df_dist = df[['Dist']]

    other_scaler = StandardScaler() if args.scaler == 'standard' else MinMaxScaler()
    speed_scaler = StandardScaler() if args.scaler == 'standard' else MinMaxScaler()
    dist_scaler = StandardScaler() if args.scaler == 'standard' else MinMaxScaler()

    np_others_scale = other_scaler.fit_transform(df_others)
    np_speed_scale = speed_scaler.fit_transform(df_speed)
    np_dist_scale = dist_scaler.fit_transform(df_dist)

    return np_speed_scale, np_dist_scale, np_others_scale, speed_scaler, dist_scaler, other_scaler

def prep_data(df, train_data_file, val_data_file, test_data_file, ref_time, pred_time):

    np_speed_scale, np_dist_scale, np_others_scale, _, _, _ = scale_data(df)
    df_others_scale = pd.DataFrame(data=np_others_scale, columns=['Heading', 'Ax', 'Ay', 'Yawrate', 'RadiusOfCurve'])
    df_speed_scale = pd.DataFrame(data=np_speed_scale, columns=['Speed'])
    df_dist_scale = pd.DataFrame(data=np_dist_scale, columns=['Dist'])

    df_spat = pd.get_dummies(df['Status'])
    df_spat.columns = ['red', 'green', 'yellow']
    df = pd.concat([df, df_spat], axis=1)
    df_no_scale = df[['red', 'green', 'yellow', 'Instance']]

    df_scaled = pd.concat([df_speed_scale, df_dist_scale, df_others_scale, df_no_scale], axis=1)

    all_traj = [x for _, x in df_scaled.groupby('Instance')]
    random.shuffle(all_traj)
    data_, test_data = all_traj[:-args.n_test_data], all_traj[-args.n_test_data:]

    all_data = []
    for instance in data_:
        instance_np = instance.values
        for i in range(int(len(instance_np)/(10 * ref_time))):
            try:
                rand_n = random.randint(10 * ref_time, len(instance_np) - 10 * pred_time)
            except ValueError:
                continue
            x = instance_np[rand_n - 10 * ref_time:rand_n, :-1] # all column except Status
            y_ = instance_np[rand_n:rand_n + 10 * pred_time, 0] # the Speed column
            y = y_[::int(args.pred_time * 10 / args.points)] # just predict 10 points
            all_data.append([x, y])

    random.shuffle(all_data)
    train_data = all_data[: int(args.train_ratio * len(all_data))]
    val_data = all_data[int(args.train_ratio * len(all_data)):]


    # Save the preprocessed data to file
    with open(train_data_file, 'wb') as fp:
        pickle.dump(train_data, fp)
    with open(val_data_file, 'wb') as fp:
        pickle.dump(val_data, fp)
    with open(test_data_file, 'wb') as fp:
        pickle.dump(test_data, fp)

    return train_data, val_data, test_data

if __name__ == '__main__':
    prep_data(df=pd.read_csv(args.all_data_file, index_col=False),
              train_data_file=args.train_data_file,
              val_data_file=args.val_data_file,
              test_data_file=args.test_data_file,
              ref_time=args.ref_time,
              pred_time=args.pred_time)