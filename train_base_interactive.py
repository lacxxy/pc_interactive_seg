"""
Author: Benny
Date: Nov 2019
"""
## salloc -N 1 --ntasks 10 --account=lh --gpus=1
import argparse
import os
from S3DISDataLoader import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from scipy.spatial.distance import cdist  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['background', 'target']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--round', type=int, default=1, help='interactivate round')

    return parser.parse_args()

def generate_fnfp(pred_choice,batch_label,batch_size):
    # 将 flat arrays 重塑回原来的批次形状
    pred_choice_reshaped = pred_choice.reshape(batch_size, -1)
    batch_label_reshaped = batch_label.reshape(batch_size, -1)

    # 计算 FP 和 FN
    # FP: 预测为正类，但实际上不是正类（即预测错误）
    # FN: 预测为负类，但实际上不是负类（即预测错误）
    fp = (pred_choice_reshaped == 1) & (batch_label_reshaped == 0)
    fn = (pred_choice_reshaped == 0) & (batch_label_reshaped == 1)

    # 将布尔值转换为整数，True 转换为 1，False 转换为 0
    fp = fp.astype(int)
    fn = fn.astype(int)

    # 输出结果检查
    # print("FP shape:", np.shape(fp))# (16, 4096)
    # print("FN shape:", np.shape(fn))# (16, 4096)
    return fn,fp

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = '/public/home/lh/lh/czh/stanford_indoor3d'
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader) * args.round
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        iou_sum = [0 for _ in range(args.round)]
        iou_test_sum = [0 for _ in range(args.round)]
        classifier = classifier.train()

        for i, (points, target,click_set) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            origin_points = points
            positive_click = click_set
            negative_click = np.zeros((BATCH_SIZE, 0, 3))
            for interactive_round in range(args.round):
                optimizer.zero_grad()
                points = origin_points
                if interactive_round == 0:
                    points = points.data.numpy()
                else:
                    #计算上一轮的fn，选择一个点
                    points = points.data.numpy()
                    # 初始化结果数组
                    selected_po_points = np.zeros((fn.shape[0], 3))
                    selected_na_points = np.zeros((fn.shape[0], 3))

                    # 遍历每个 batch
                    for i in range(fn.shape[0]):
                        fn_indices = np.where(fn[i] == 1)[0]
                        fp_indices = np.where(fp[i] == 1)[0]
                        
                        if len(fn_indices) > len(fp_indices):
                            random_index = np.random.choice(fn_indices)
                            selected_po_points[i] = points[i, random_index, :3]
                        elif len(fn_indices) < len(fp_indices):
                            random_index = np.random.choice(fp_indices)
                            selected_na_points[i] = points[i, random_index, :3]

                    # 将选定的点添加到点击集合中
                    selected_po_points = selected_po_points[:, np.newaxis, :]
                    selected_na_points = selected_na_points[:, np.newaxis, :]
                    
                    positive_click = np.concatenate((positive_click, selected_po_points), axis=1)# (64, 2, 3)
                    negative_click = np.concatenate((negative_click, selected_na_points), axis=1)# (64, 1, 3)

                    # 更新 points 的第 9 和第 10 列
                    min_distances_po = np.zeros((BATCH_SIZE, NUM_POINT))
                    for i in range(positive_click.shape[0]):
                        # 排除(0,0,0)点
                        valid_clicks = positive_click[i][~np.all(positive_click[i] == 0, axis=1)]
                        
                        if valid_clicks.size > 0:
                            distances = cdist(points[i, :, :3], valid_clicks, 'euclidean')
                            min_distances_po[i] = np.min(distances, axis=1)
                        else:
                            # 如果没有有效的点击点，可以设置最小距离为某个默认值或保持全0
                            min_distances_po[i] = np.full(NUM_POINT, 0)  # 或者其他默认值

                    # 将最小距离值写入points数组
                    points[:, :, 9] = min_distances_po

                    min_distances_na = np.zeros((BATCH_SIZE, NUM_POINT))
                    for i in range(negative_click.shape[0]):
                        # 排除(0,0,0)点
                        valid_clicks = negative_click[i][~np.all(negative_click[i] == 0, axis=1)]
                        #print(valid_clicks)
                        if valid_clicks.size > 0:
                            distances = cdist(points[i, :, :3], valid_clicks, 'euclidean')
                            min_distances_na[i] = np.min(distances, axis=1)
                        else:
                            min_distances_na[i] = np.full(NUM_POINT, 0) 
                    points[:, :, 10] = min_distances_na

                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss.backward()
                optimizer.step()

                pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                fn,fp = generate_fnfp(pred_choice,batch_label,BATCH_SIZE)# (16, 4096)

                correct = np.sum(pred_choice == batch_label)
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                loss_sum += loss 


                total_correct_class = np.sum((pred_choice == 1) & (batch_label == 1))
                total_iou_deno_class = np.sum(((pred_choice == 1) | (batch_label == 1)))
                iou_sum[interactive_round] += total_correct_class / total_iou_deno_class
                
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))     
        iou_per_class_str = '------- IoU --------\n'
        for l in range(args.round):
            iou_per_class_str += 'round %s, IoU: %.3f \n' % (
                l,
                iou_sum[l]/len(trainDataLoader)
            )
        log_string(iou_per_class_str)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader) * args.round
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target,click_set) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                origin_points = points
                positive_click = click_set
                negative_click = np.zeros((BATCH_SIZE, 0, 3))
                #print(np.shape(target))#([16, 4096])
                #print(np.shape(click_set))#([16,1,3])

                for interactive_round in range(args.round):
                    points = origin_points
                    if interactive_round == 0:
                        points = points.data.numpy()
                    else:
                        #计算上一轮的fn，选择一个点
                        points = points.data.numpy()
                        # 初始化结果数组
                        selected_po_points = np.zeros((fn.shape[0], 3))
                        selected_na_points = np.zeros((fn.shape[0], 3))

                        # 遍历每个 batch
                        for i in range(fn.shape[0]):
                            fn_indices = np.where(fn[i] == 1)[0]
                            fp_indices = np.where(fp[i] == 1)[0]
                            if len(fn_indices) > len(fp_indices):
                                random_index = np.random.choice(fn_indices)
                                selected_po_points[i] = points[i, random_index, :3]
                            elif len(fn_indices) < len(fp_indices):
                                random_index = np.random.choice(fp_indices)
                                selected_na_points[i] = points[i, random_index, :3]

                        # 将选定的点添加到点击集合中
                        selected_po_points = selected_po_points[:, np.newaxis, :]
                        selected_na_points = selected_na_points[:, np.newaxis, :]
                        
                        positive_click = np.concatenate((positive_click, selected_po_points), axis=1)# (64, 2, 3)
                        negative_click = np.concatenate((negative_click, selected_na_points), axis=1)# (64, 1, 3)

                        # 更新 points 的第 9 和第 10 列
                        min_distances_po = np.zeros((BATCH_SIZE, NUM_POINT))
                        for i in range(positive_click.shape[0]):
                            # 排除(0,0,0)点
                            valid_clicks = positive_click[i][~np.all(positive_click[i] == 0, axis=1)]
                            
                            if valid_clicks.size > 0:
                                distances = cdist(points[i, :, :3], valid_clicks, 'euclidean')
                                min_distances_po[i] = np.min(distances, axis=1)
                            else:
                                # 如果没有有效的点击点，可以设置最小距离为某个默认值或保持全0
                                min_distances_po[i] = np.full(NUM_POINT, 0)  # 或者其他默认值

                        # 将最小距离值写入points数组
                        points[:, :, 9] = min_distances_po

                        min_distances_na = np.zeros((BATCH_SIZE, NUM_POINT))
                        for i in range(negative_click.shape[0]):
                            # 排除(0,0,0)点
                            valid_clicks = negative_click[i][~np.all(negative_click[i] == 0, axis=1)]
                            #print(valid_clicks)
                            if valid_clicks.size > 0:
                                distances = cdist(points[i, :, :3], valid_clicks, 'euclidean')
                                min_distances_na[i] = np.min(distances, axis=1)
                            else:
                                min_distances_na[i] = np.full(NUM_POINT, 0) 
                        points[:, :, 10] = min_distances_na
            
                    points = torch.Tensor(points)
                    points, target = points.float().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)

                    seg_pred, trans_feat = classifier(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                    batch_label = target.cpu().data.numpy()
                    target = target.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred, target, trans_feat, weights)
                    loss_sum += loss
                    pred_val = np.argmax(pred_val, 2)
                    #correct = np.sum((pred_val == batch_label))

                    # total_correct += correct
                    total_seen += (BATCH_SIZE * NUM_POINT)
                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                    labelweights += tmp

                    batch_label = batch_label.reshape(-1, NUM_POINT)

                    pred_choice = (seg_pred.cpu().data.max(1)[1].numpy()).reshape(-1,NUM_POINT)
                    
                    correct = np.sum((pred_choice == batch_label))
                    total_correct += correct

                    fn,fp = generate_fnfp(pred_choice,batch_label,BATCH_SIZE)# (16, 4096)

                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_choice == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_choice == l) | (batch_label == l)))

                        total_correct_class_test = np.sum((pred_choice == 1) & (batch_label == 1))
                        total_iou_deno_class_test = np.sum(((pred_choice == 1) | (batch_label == 1)))


                    iou_test_sum[interactive_round] += total_correct_class_test / total_iou_deno_class_test

            labelweights = labelweights.astype(float) / np.sum(labelweights.astype(float))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            iou_test_per_round_str = '------- IoU --------\n'
            for l in range(args.round):
                iou_test_per_round_str += 'round %s, IoU: %.3f \n' % (
                    l,
                    iou_test_sum[l]/len(testDataLoader)
                )
            log_string(iou_test_per_round_str)

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
