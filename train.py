import torch
import config
args = config.config()
import random
import matplotlib.pyplot as plt
import numpy as np



def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src, trg = batch
        src = src.transpose(0, 1).float().to(args.device)
        trg = trg.transpose(0, 1).float().to(args.device)

        optimizer.zero_grad()
        output = model(src, trg, args.teacher_forcing_ratio)
        loss = 10000 * (criterion(output.view(args.points, args.batch_size).float(), trg))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, trg = batch
            src = src.transpose(0, 1).float().to(args.device)
            trg = trg.transpose(0, 1).float().to(args.device)
            output = model(src, trg, args.teacher_forcing_ratio)
            loss =  10000 * (criterion(output.view(args.points, args.batch_size).float(), trg))
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def show_predictiopn(test_data, model, speed_scaler, dist_scaler):
    if not args.load_weights:
        print('Set load weights in config')
        exit()
    # prepare for the text data
    for instance_df in test_data:
        if instance_df.shape[0] <= (args.ref_time + args.pred_time) * 10:
            print('Trajectory too short')
            continue
        instance_np = instance_df.values

        x_np_test = instance_np[:, :args.x_dim]
        speed_test = instance_np[:, 0]
        dist_test = instance_np[:, 1]

        rand_position = random.randint(args.ref_time * 10, len(x_np_test) - args.pred_time * 10)

        dist_select_gt = dist_test[rand_position:rand_position + args.pred_time * 10]
        dist_select_gt = dist_select_gt[::int(args.pred_time * 10 / args.points)]

        dist_select_ref = dist_test[rand_position - args.ref_time * 10:rand_position]
        speed_select_ref = speed_test[rand_position - args.ref_time * 10:rand_position]

        x_test = x_np_test[rand_position - args.ref_time * 10:rand_position, :]
        y_test = speed_test[rand_position:rand_position + args.pred_time * 10]
        ground_truth = y_test[::int(args.pred_time * 10 / args.points)]

        test_input_tensor = torch.Tensor(x_test).unsqueeze(0).transpose(0, 1).to(args.device)
        test_target_tensor = torch.Tensor(ground_truth).unsqueeze(0).transpose(0, 1).to(args.device)

        model.eval()
        with torch.no_grad():
            pred_speed = model(test_input_tensor, test_target_tensor, args.teacher_forcing_ratio)
            pred_speed = np.squeeze(pred_speed.cpu().numpy())


        plt.figure()
        plt.xlabel('Distance to stop line (meter)')
        plt.ylabel('Vehicle speed (m/s)')
        plt.grid(True)
        plt.title('Speed prediction')
        plt.plot(dist_scaler.inverse_transform(dist_test.reshape(-1, 1)),
                 speed_scaler.inverse_transform(speed_test.reshape(-1, 1)),
                 c='y',
                 linestyle='dashed', label='trajectory')
        plt.plot(dist_scaler.inverse_transform(dist_select_ref.reshape(-1, 1)),
                 speed_scaler.inverse_transform(speed_select_ref.reshape(-1, 1)),
                 c='g', linestyle='dashed', label='Reference points')
        plt.scatter(dist_scaler.inverse_transform(dist_select_gt.reshape(-1, 1)),
                    speed_scaler.inverse_transform(ground_truth.reshape(-1, 1)),
                    c='g',
                    label='Ground truth')
        plt.plot(dist_scaler.inverse_transform(dist_select_gt.reshape(-1, 1)),
                 speed_scaler.inverse_transform(pred_speed.reshape(-1, 1)),
                 c='r',
                 label='Prediction')
        plt.legend()
        plt.show()