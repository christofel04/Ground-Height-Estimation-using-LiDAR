#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



import argparse
import os
import shutil
import yaml
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# from modules import gnd_est_Loss
from model import GroundEstimatorNet
from modules.loss_func import MaskedHuberLoss,SpatialSmoothLoss
from dataset_utils.dataset_provider import get_train_loader, get_valid_loader
from utils.point_cloud_ops import points_to_voxel
import ipdb as pdb

use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used






#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################


parser = argparse.ArgumentParser()
parser.add_argument('--print-freq', '-p', default=100, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--config', default='config/config_kittiSem.yaml', type=str, metavar='PATH', help='path to config file (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('-s', '--save_checkpoints', dest='save_checkpoints', action='store_true',help='evaluate model on validation set')
parser.add_argument('--start_epoch', default=0, type=int, help='epoch number to start from')
args = parser.parse_args()

if os.path.isfile(args.config):
    print("using config file:", args.config)
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    class ConfigClass:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = ConfigClass(**config_dict) # convert python dict to class for ease of use

else:
    print("=> no config file found at '{}'".format(args.config))

#############################################xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx#######################################




#train_loader =  get_train_loader(cfg.data_dir, cfg.batch_size, skip = 4)
#valid_loader =  get_valid_loader(cfg.data_dir, cfg.batch_size, skip = 6)

train_loader =  get_train_loader(cfg.data_train_dir, cfg.batch_size, skip = 1 , is_argoverse_road_detection= True , num_points= cfg.num_points , max_number_of_log = cfg.data_train_max_number_of_log )
print( "Is evaluate DA evaluation : " + str( cfg.is_DA_evaluation_eval ))
valid_loader =  get_valid_loader(cfg.data_val_dir, batch =( 1 if cfg.is_DA_evaluation_eval==True else cfg.batch_size ) , skip = 1 , is_argoverse_road_detection= True , num_points = cfg.num_points, LIST_OF_LOG_FOR_VISUALIZATION = cfg.LIST_OF_LOG_FOR_VISUALIZATION , is_DA_BEV_evaluation = cfg.is_DA_evaluation_eval )

model = GroundEstimatorNet(cfg).cuda()
optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

lossHuber = nn.SmoothL1Loss(reduction = "mean").cuda()
lossSpatial = SpatialSmoothLoss().cuda()

#is_DA_visualization_eval : bool = True 

def train(epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # top1 = AverageMeter()

    # switch to train mode
    model.train()
    start = time.time()

    for batch_idx, (data, labels) in enumerate(train_loader):

        if cfg.is_DA_evaluation_eval == True :

            labels = labels[0]
            
        labels = torch.nan_to_num( labels , nan = -100 , posinf = 100 , neginf = -100 )
        # Clip the label so the highest point is 10 m ,lowest point is -10 m
        labels = torch.clip(labels , min= -10 , max=10 )
        #print( "Ground height estimation before normalization is : " + str( labels ))
        # Normalizes labels 
        labels = ( labels + 10 )/20
        #print( "Ground Height estimation after normalization is : " + str( labels ))

        data_time.update(time.time() - start) # measure data loading time
        B = data.shape[0] # Batch size
        N = data.shape[1] # Num of points in PointCloud

        voxels = []; coors = []; num_points = []; mask = []
        # kernel = np.ones((3,3),np.uint8)

        data = data.numpy()
        for i in range(B):
            v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
            # m = np.zeros((100,100),np.uint8)
            # ind = c[:,1:]
            # m[tuple(ind.T)] = 1
            # m = cv2.dilate(m,kernel,iterations = 1)

            c = torch.from_numpy(c)
            c = F.pad(c, (1,0), 'constant', i)
            voxels.append(torch.from_numpy(v))
            coors.append(c)
            num_points.append(torch.from_numpy(n))
            # mask.append(torch.from_numpy(m))
# 
        voxels = torch.cat(voxels).float().cuda()
        coors = torch.cat(coors).float().cuda()
        num_points = torch.cat(num_points).float().cuda()
        labels = labels.float().cuda()
        # mask = torch.stack(mask).cuda()
        
        #print( "Ground Height target of DA detection is : " + str( labels ))

        optimizer.zero_grad()

        output = model(voxels, coors, num_points)
        # pdb.set_trace()

        #print( "Dimension of ground height estimation is : {} and ground height labels are : {}".format( output.shape , labels.shape ))
        
        # Replaced all infinites with flag values
        
        

        loss = cfg.alpha * lossHuber(output, labels) + cfg.beta * lossSpatial(output)
        
        #print( "Labels of ground height estimation is : " + str( labels ))
        # loss = lossHuber(output, labels)
        # loss = masked_huber_loss(output, labels, mask)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)

        optimizer.step() # optimiser step must be after clipping bcoz optimiser step updates the gradients.

        losses.update(loss.item(), B)
        
        #print( "Loss of the batch is : " + str( loss.item()))

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()


        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return losses.avg



def validate( epoch : int = None ):

    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    assert os.path.exists( cfg.DA_evaluation_result_folder )

    # switch to evaluate mode
    model.eval()
    # if args.evaluate:
    #     model.train()
    with torch.no_grad():
        start = time.time()
        
        sum_of_accuracy_DA_detection = 0
        
        sum_of_f1_score_DA_detection = 0

        list_of_visualized_prediction_image = []

        for batch_idx, (data, labels) in enumerate(valid_loader):
        

            #if valid_loader.dataset.is_DA_BEV_evaluation == True :
            
            #    labels	
            
            data_time.update(time.time() - start) # measure data loading time
            
            if cfg.is_DA_evaluation_eval == True :

                DA_rasterized_label = labels[1]
                
                lidar_pts_visualization_this_epoch = labels[2]
                
                labels = labels[0]
                
                #print( "Shape of Label is : " + str( labels.shape ) + "with labels are " + str( ))
            	
            labels = torch.nan_to_num( labels , nan = -100 , posinf = 100 , neginf = -100 )
            # Clip the label with minimum height = -10 and maximum height = 10
            labels = torch.clip( labels , min = -10 , max = 10 )
            # Convert ground height ground truth to initial ground height
            labels = ( labels + 10 )/20 #200 * labels - 100 
            assert labels.max() <= 1
            #print( "Ground height prediction is : " + str( labels ))
            B = data.shape[0] # Batch size
            N = data.shape[1] # Num of points in PointCloud

            voxels = []; coors = []; num_points = []; mask = []
            # kernel = np.ones((3,3),np.uint8)

            data = data.numpy()
            for i in range(B):
                v, c, n = points_to_voxel(data[i], cfg.voxel_size, cfg.pc_range, cfg.max_points_voxel, True, cfg.max_voxels)
                # m = np.zeros((100,100),np.uint8)
                # ind = c[:,1:]
                # m[tuple(ind.T)] = 1
                # m = cv2.dilate(m,kernel,iterations = 1)

                c = torch.from_numpy(c)
                c = F.pad(c, (1,0), 'constant', i)
                voxels.append(torch.from_numpy(v))
                coors.append(c)
                num_points.append(torch.from_numpy(n))
                # mask.append(torch.from_numpy(m))

            voxels = torch.cat(voxels).float().cuda()
            coors = torch.cat(coors).float().cuda()
            num_points = torch.cat(num_points).float().cuda()
            labels = labels.float().cuda()
            # mask = torch.stack(mask).cuda()

            optimizer.zero_grad()

            output = model(voxels, coors, num_points)
            # pdb.set_trace()

            loss = cfg.alpha * lossHuber(output, labels) + cfg.beta * lossSpatial(output)
            # loss = lossHuber(output, labels)
            # loss = masked_huber_loss(output, labels, mask)

            losses.update(loss.item(), B)

            # measure elapsed time
            batch_time.update(time.time() - start)
            
            if cfg.is_DA_evaluation_eval == True :
            
                # Convert back normalized ground height estimation to height
                print( "Output of the Ground Height Estimation is : " + str( output ))
                output = output[0]
                
                output = output*20 - 10
                
                print( "Prediction of Ground Height Estimation after convert is : " + str( output ))
                
                labels = labels * 20 - 10
                
                print( "Height of ground height estimation is : " + str( labels ))
                
                output_DA_label = convert_ground_estimation_height_to_DA_prediction( output , known_ground_height_treshold = 0.5)
                
                #print( "Output DA Label is : " + str( output_DA_label ))

                #print( "Shape of confidence label is : " + str( conf_label ) + "with average drivable area : " + str( np.mean( conf_label )))

                conf_label = torch.squeeze( DA_rasterized_label )
                
                print( "Labels of DA detection : " + str( conf_label ))

                height_of_image , width_of_image = conf_label.shape

                #print( "Heigh of the image is : {} and width : {}".format( height_of_image , width_of_image ))

                #conf_label = conf_label.reshape( -1 , height_of_image , width_of_image )

                road_detection_label = np.array( conf_label )#.astype(np.bool)

                road_detection_prediction = np.array( output_DA_label )#.astype(np.bool)
                
                assert road_detection_label.shape == road_detection_prediction.shape
                
                # Convert ground height estimation to have same direction with drivable label
                
                road_detection_prediction_image = Image.fromarray( road_detection_prediction )
                
                road_detection_prediction_image_with_same_orientation = road_detection_prediction_image.rotate(270).transpose( Image.FLIP_LEFT_RIGHT ).rotate(180)
                
                road_detection_prediction = np.array( road_detection_prediction_image_with_same_orientation )
                
                
                # Find F1 Accuracy and F1 score of point- wise DA detection
                
                accuracy = accuracy_score( road_detection_label.astype( np.bool ) , road_detection_prediction.astype( np.bool ) )
                
                f1_score_prediction = f1_score( road_detection_label , road_detection_prediction , average = "micro")
                
                print( "Accuracy score of GndNet prediction is : {} and F1- Score : {}".format( accuracy , f1_score_prediction ))
                
                sum_of_accuracy_DA_detection = sum_of_accuracy_DA_detection + accuracy
                
                sum_of_f1_score_DA_detection = sum_of_f1_score_DA_detection + f1_score_prediction
                
                

                # Change image to RGB image with red color as drivable area

                road_detection_label_with_color = np.array( [[[0, 255 , 0] if i == 1 else [ 255 , 255 , 255 ] for i in j ] for j in road_detection_label] ).astype(np.uint8)

                road_detection_prediction_with_color = np.array( [[[120, 120  , 120 ] if i == 1 else [255,255,255] for i in j] for j in road_detection_prediction] ).astype(np.uint8)
                
                # Make lidar points segmentation for point-wise DA detection
                
                lidar_pts_visualization_this_epoch = np.array( torch.squeeze( lidar_pts_visualization_this_epoch )) #lidar_pts_visualization( lidar_pts )
                
                #print( "Number of voxel with lidar point cloud is : " + str( np.sum()))
               
                
                lidar_point_visualization_with_DA_label_color = lidar_pts_visualization_this_epoch.astype( np.uint8 ) # torch.squeeze( lidar_pts_visualization_this_epoch )#.copy()
                
                lidar_point_visualization_with_DA_label_color_BEV_image = Image.fromarray( lidar_point_visualization_with_DA_label_color ).rotate( 90 )
                
                lidar_point_visualization_with_DA_label_color = np.array( lidar_point_visualization_with_DA_label_color_BEV_image  )
                
                assert lidar_point_visualization_with_DA_label_color.shape[ : 2 ] == road_detection_prediction.shape[ : 2 ] , "Shape of lidar point visualization and road detection prediction have to be same that is shape of lidar point visualization is : " + str( lidar_point_visualization_with_DA_label_color.shape ) + " and shape of road detection prediction is : " + str( road_detection_prediction.shape ) 
                
                for x_lidar_point_visualization in range(lidar_pts_visualization_this_epoch.shape[0]) :
                
                    for y_lidar_point_visualization in range( lidar_pts_visualization_this_epoch.shape[1]) :
                    
                        if lidar_pts_visualization_this_epoch[ x_lidar_point_visualization ][ y_lidar_point_visualization ][0] == 0  :
                        
                            #print( "Find a DA voxel with raw LiDAR point in the BEV voxel" )
                            
                            if road_detection_prediction[ x_lidar_point_visualization ][ y_lidar_point_visualization ] == 1 :
                            
                            
                            
                                lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 120 , 120 , 120 ]
                                
                                
                            else :
                            
                                lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 0, 0 , 255 ]
                            
                                                            
                        else :
                        
                                lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [255, 255, 255 ]                
                            
                            
                        
                        
                    
                    
                
                
                
                

                # Make point-wise DA detection ground truth
                lidar_point_visualization_with_DA_label_ground_truth_color = lidar_pts_visualization_this_epoch.astype( np.uint8 ) # torch.squeeze( lidar_pts_visualization_this_epoch )#.copy()
                
                lidar_point_visualization_with_DA_label_ground_truth_color = Image.fromarray( lidar_point_visualization_with_DA_label_ground_truth_color ).rotate( 90 )
                
                lidar_point_visualization_with_DA_label_ground_truth_color_image = np.array( lidar_point_visualization_with_DA_label_ground_truth_color  )
                
                assert lidar_point_visualization_with_DA_label_ground_truth_color_image.shape[ : 2 ] == road_detection_label.shape[ : 2 ] , "Shape of lidar point visualization and road detection prediction have to be same that is shape of lidar point visualization is : " + str( lidar_point_visualization_with_DA_label_ground_truth_color_image.shape ) + " and shape of road detection prediction is : " + str( road_detection_label.shape ) 
                
                for x_lidar_point_visualization in range(lidar_pts_visualization_this_epoch.shape[0]) :
                
                    for y_lidar_point_visualization in range( lidar_pts_visualization_this_epoch.shape[1]) :
                    
                        if lidar_pts_visualization_this_epoch[ x_lidar_point_visualization ][ y_lidar_point_visualization ][0] == 0  :
                        
                            #print( "Find a DA voxel with raw LiDAR point in the BEV voxel" )
                            
                            if road_detection_label[ x_lidar_point_visualization ][ y_lidar_point_visualization ] == 1 :
                            
                            
                            
                                lidar_point_visualization_with_DA_label_ground_truth_color_image[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 120 , 120 , 120 ]
                                
                                
                            else :
                            
                                lidar_point_visualization_with_DA_label_ground_truth_color_image[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 0, 0 , 255 ]
                            
                                                            
                        else :
                        
                                lidar_point_visualization_with_DA_label_ground_truth_color_image[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [255, 255, 255 ]                
                            
                            
                            #print( "Shape of road detection label is : " + str( road_detection_label_with_color.shape ) + " and shape of road detection prediction is : " + str( road_detection_prediction_with_color.shape ))

                # Make LiDAR point visualization on DA ground truth label
                
                lidar_pts_visualization_on_DA_ground_truth_label = lidar_pts_visualization_this_epoch.copy()
                
                for x_lidar_point_visualization in range(lidar_pts_visualization_this_epoch.shape[0]) :
                
                    for y_lidar_point_visualization in range( lidar_pts_visualization_this_epoch.shape[1]) :
                    
                        if lidar_pts_visualization_this_epoch[ x_lidar_point_visualization ][ y_lidar_point_visualization ][0] == 0  :
                        
                            #print( "Find a DA voxel with raw LiDAR point in the BEV voxel" )
                            
                            lidar_pts_visualization_on_DA_ground_truth_label[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 0 , 0 , 255 ]
                            
                            #if road_detection_label[ x_lidar_point_visualization ][ y_lidar_point_visualization ] == 1 :

                        elif road_detection_label_with_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ][0] == 0  :
                
                             lidar_pts_visualization_on_DA_ground_truth_label[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 120 , 120 , 120 ]               
                
                
                #lidar_points_data = torch.squeeze( data[ "lidar_data" ] )

                #lidar_points_data = torch.concatenate( [ lidar_points_data[ : , 0 ]* ( 1 / self.cfg.list_grid_xy[0] ) , lidar_points_data[ : , 1 ]* ( 1/ self.cfg.list_grid_xy[1] ) , lidar_points_data[ : ,2 : ]], dim= 1 )

                #print( "Shape of lidar points is : " + str( lidar_points_data.shape ))

                """
                for i, lidar_point in enumerate( torch.squeeze( lidar_points_data )) :

                    print( "Lidar point is : " + str( lidar_point ))

                    if (( lidar_point[0] >= -width_of_image/2 ) and
                        ( lidar_point[0] <= width_of_image/2 ) and
                        ( lidar_point[1] >= -height_of_image*5/12) and 
                        ( lidar_point[1] <= height_of_image*7/12 )) : 

                        #print( "Writes lidar point number : " + str( i )) 

                        road_detection_label_with_color = cv2.circle(road_detection_label_with_color,  ( int( (lidar_point[0] + width_of_image/2)), int( (lidar_point[1] + height_of_image*7/12))), radius_lidar_point, ( 0 , 0 , int(100 + ( -lidar_point[2] - 10 )* 3 )))

                        road_detection_prediction_with_color = cv2.circle(road_detection_prediction_with_color, ( int( (lidar_point[0] + width_of_image/2)), int( (lidar_point[1] + height_of_image*7/12))), radius_lidar_point, ( 0 , 0 , int( 100 + ( -lidar_point[2] - 10 )* 3 )))
                """

                #resize, first image
                image1 = Image.fromarray( road_detection_label_with_color ).resize((1000 , 1200))
                image2 = Image.fromarray( road_detection_prediction_with_color ).resize( ( 1000,1200 ))
                print( "Size of prediction image is : " + str( np.array( image2 ).shape ))
                image1_size = image1.size
                image2_size = image2.size
                new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                new_image.paste(image1,(0,0))
                new_image.paste(image2,( image1_size[0], 0 ))
                
                lidar_point_visualization_with_DA_label_color = np.array( lidar_point_visualization_with_DA_label_color ).astype( np.uint8 )
                lidar_point_visualization_with_color = Image.fromarray( lidar_point_visualization_with_DA_label_color ).resize(( 1000 , 1200 ))
                # Save file to external file
                lidar_point_visualization_with_color.save( "Visualization_GndNet_for_DA_Detection_epoch_{}_prediction_{}.png".format( epoch, batch_idx) , "PNG")
                image1 = new_image.copy()
                
                image1_size = image1.size
                image2_size = lidar_point_visualization_with_color.size
                new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                new_image.paste(image1,(0,0))
                new_image.paste(lidar_point_visualization_with_color,( image1_size[0], 0 ))

                lidar_point_visualization_with_DA_label_color = np.array( lidar_point_visualization_with_DA_label_ground_truth_color_image ).astype( np.uint8 )
                lidar_point_visualization_with_color = Image.fromarray( lidar_point_visualization_with_DA_label_color ).resize(( 1000 , 1200 ))
                # Save file to external file
                lidar_point_visualization_with_color.save( "Visualization_Ground_Truth_GndNet_for_DA_Detection_epoch_{}_prediction_{}.png".format( epoch, batch_idx) , "PNG")
                image1 = new_image.copy()
                
                image1_size = image1.size
                image2_size = lidar_point_visualization_with_color.size
                new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                new_image.paste(image1,(0,0))
                new_image.paste(lidar_point_visualization_with_color,( image1_size[0], 0 ))
                #list_of_visualized_prediction_image.append( np.array( new_image ) )
                
                lidar_point_visualization_with_DA_label_color = np.array( lidar_pts_visualization_on_DA_ground_truth_label ).astype( np.uint8 )
                lidar_point_visualization_with_color = Image.fromarray( lidar_point_visualization_with_DA_label_color ).resize(( 1000 , 1200 ))
                # Save file to external file
                lidar_point_visualization_with_color.save( "Visualization_Ground_Truth_GndNet_using_LiDAR_for_DA_Detection_epoch_{}_prediction_{}.png".format( epoch, batch_idx) , "PNG")
                image1 = new_image.copy()
                
                image1_size = image1.size
                image2_size = lidar_point_visualization_with_color.size
                new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                new_image.paste(image1,(0,0))
                new_image.paste(lidar_point_visualization_with_color,( image1_size[0], 0 ))
                #list_of_visualized_prediction_image.append( np.array( new_image ) )
                
                lidar_point_visualization_with_DA_label_color = np.array( lidar_pts_visualization_this_epoch ).astype( np.uint8 )
                lidar_point_visualization_with_color = Image.fromarray( lidar_point_visualization_with_DA_label_color ).resize(( 1000 , 1200 ))
                # Save file to external file
                lidar_point_visualization_with_color.save( "Visualization_LiDAR_Point_this_Epoch_GndNet_using_LiDAR_for_DA_Detection_epoch_{}_prediction_{}.png".format( epoch, batch_idx) , "PNG")
                image1 = new_image.copy()
                
                image1_size = image1.size
                image2_size = lidar_point_visualization_with_color.size
                new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                new_image.paste(image1,(0,0))
                new_image.paste(lidar_point_visualization_with_color,( image1_size[0], 0 ))
                list_of_visualized_prediction_image.append( np.array( new_image ) )
            	
            	
            
            
            start = time.time()

		

            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       batch_idx, len(valid_loader), batch_time=batch_time, loss=losses))

    # Save visualized validation to folder
                            
    if ( cfg.is_DA_evaluation_eval == True ) :

        if epoch is None :

            epoch = "_"

        # Visualizing some Lane Detection dataset

        sns.set_theme()

        f, axarr = plt.subplots( len( list_of_visualized_prediction_image ), figsize = ( 20 , 30 ))
        
        plt.title( "Visualization of Road Detection Epoch : {}".format( epoch ))
        plt.axis('off')

        for i in range( len( list_of_visualized_prediction_image) ):

            axarr[ i ].imshow(  list_of_visualized_prediction_image[i] )
            axarr[ i ].set_title( "Lane Image Data Label and Prediction Data : " + str(i)) 
            axarr[ i ].set_axis_off()
            

            f.tight_layout()
            #plt.show()

        if cfg.is_save_DA_evaluation_eval == True : 
        
            plt.savefig( cfg.DA_evaluation_result_folder + "visualization_epoch_" + str( epoch ) + ".png" )
            print( "Save image visualization to : " + str( cfg.DA_evaluation_result_folder + "visualization_epoch_" + str( epoch ) + ".png"   ))

	
        print("Average accuracy of DA detection using GndNet is : {} and average F1- score is : {} with sum of accuracies : {} and sum of F1 score : {}".format( sum_of_accuracy_DA_detection / len( valid_loader ) , sum_of_f1_score_DA_detection/ len( valid_loader ) , sum_of_accuracy_DA_detection , sum_of_f1_score_DA_detection))
        
        #return plt
        
		
    return losses.avg

def convert_ground_estimation_height_to_DA_prediction( ground_height_estimation_result : np.array , known_ground_height_treshold : float = -1.5 ) :

    #print( "Shape of ground height prediction is : " + str( ground_height_estimation_result.shape ))
    
    ground_height_estimation_result = torch.squeeze( ground_height_estimation_result )
    
    #print( "Shape of ground height prediction after squeezing is : " + str( ground_height_estimation_result.shape ))
   
   
    
    da_drivable_prediction = np.zeros( ground_height_estimation_result.shape ) 

    for x_ground_height_estimation in range( ground_height_estimation_result.shape[-2] ) :

        for y_ground_height_estimation in range( ground_height_estimation_result.shape[ -1 ] ) :

            if ground_height_estimation_result[ x_ground_height_estimation ][ y_ground_height_estimation ] <= known_ground_height_treshold :

                da_drivable_prediction[ x_ground_height_estimation ][ y_ground_height_estimation ] = 1

    return da_drivable_prediction


def lidar_pts_visualization( lidar_pts , voxel_size = [ 0.2 , 0.2 ] , lidar_range = [-50 , 70 , -50 , 50 ] , HEIGHT_MEAN_TRESHOLD = -1 , HEIGHT_VARIANCE_TRESHOLD = 0.01 , DIFFERENT_MAX_MIN_HEIGHT_TRESHOLD = 0.2 , BAYESIAN_GAUSSIAN_KERNEL_RADIUS = 4 , LIDAR_POINT_FILTERING_HEIGHT_TRESHOLD = 10 ) :

    lidar_pts = lidar_pts #argoverse_data.get_lidar( index_lidar )

    lidar_range = [-50 , 70 , -50 , 50 ] 

    # Filtering the lidar height below height treshold

    lidar_pts = lidar_pts[ lidar_pts[ : , 2 ] <= LIDAR_POINT_FILTERING_HEIGHT_TRESHOLD ]

    #print( "Lidar points at beginning : " + str( lidar_pts ))
    if lidar_range[0] < 0 :
        lidar_range[1] = lidar_range[1] + -1* lidar_range[0]
        lidar_pts[ : , 0 ] = lidar_pts[ : , 0 ] + -1 * lidar_range[0]
        lidar_range[0] = 0        

        

    if lidar_range[2] < 0 :
        lidar_range[3] = lidar_range[3] + -1* lidar_range[2]
        lidar_pts[ : , 1 ] = lidar_pts[ : , 1 ] + -1* lidar_range[2]
        lidar_range[2] = 0       

    #print( "Lidar points after offset : " + str( lidar_pts ))

    
    length_of_pillar_bev = int( ( lidar_range[ 1] - lidar_range[0] )/ voxel_size[0] )
    width_of_pillar_bev = int( ( lidar_range[ 3 ] - lidar_range[2] )/ voxel_size[1] )
    
    list_of_points_in_tensor = [[[] for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]

    for lidar_point in lidar_pts :

        X_coordinate_lidar_point = int( lidar_point[0]//voxel_size[0] )

        if (( X_coordinate_lidar_point < 0 ) | ( X_coordinate_lidar_point >= length_of_pillar_bev ))  :
            continue
            
        Y_coordinate_lidar_point = int( lidar_point[1]//voxel_size[1] )

        if (( Y_coordinate_lidar_point < 0 ) | ( Y_coordinate_lidar_point >= width_of_pillar_bev )) :
            continue

        #print( "Select lidar points : " + str( lidar_point ))

        list_of_points_in_tensor[ X_coordinate_lidar_point][ Y_coordinate_lidar_point ].append( lidar_point )

    # Measure mean and height variance of every grid
        
    list_of_lidar_point_in_voxel = [[0 for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]
        
    for x in range( length_of_pillar_bev ) :

        for y in range( width_of_pillar_bev ) :

            if len( list_of_points_in_tensor[x][y] ) > 0 :

                #print( "List of points are : " + str( list_of_points_in_tensor[x][y] ))

                list_of_lidar_point_in_voxel[ x][y] = 1


    #assert None not in list_of_ground_and_none_ground_mean_and_variance_height

    list_of_lidar_point_in_voxel_with_color = [[[0 , 255 , 0] if i==1 else [ 255 , 255 , 255 ] for i in j ] for j in list_of_lidar_point_in_voxel ]

    list_of_lidar_point_in_voxel_with_color = np.array( list_of_lidar_point_in_voxel ).astype( np.uint8 )

    image_of_lidar_point_visualization = ImageOps.flip( ImageOps.mirror( Image.fromarray( list_of_lidar_point_in_voxel_with_color )))

    image_of_lidar_point_visualization_with_color = [[[0 , 0 , 255] if i==1 else [ 255 , 255 , 255 ] for i in j ] for j in np.array( image_of_lidar_point_visualization ) ]

    return np.array( image_of_lidar_point_visualization_with_color )




lowest_loss = 1

def main():
    # rospy.init_node('pcl2_pub_example', anonymous=True)
    global args, lowest_loss
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['lowest_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            

    if os.path.exists( cfg.DA_evaluation_result_folder ) == False :

        print( "Making drivable area evaluation folder in Folder : " + str( cfg.DA_evaluation_result_folder ))

        os.makedirs( cfg.DA_evaluation_result_folder )


    if args.evaluate:
        validate()
        return
        


    for epoch in range(args.start_epoch, cfg.epochs):

        # adjust_learning_rate(optimizer, epoch)
        loss_t = train(epoch)

        # evaluate on validation set
        loss_v = validate( epoch= epoch )

        scheduler.step()



        if (args.save_checkpoints):
            # remember best prec@1 and save checkpoint
            is_best = loss_v < lowest_loss
            lowest_loss = min(loss_v, lowest_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'lowest_loss': lowest_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best)




'''
Save the model for later
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
