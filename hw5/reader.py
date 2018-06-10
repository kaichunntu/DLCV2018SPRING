import numpy as np
import skvideo.io
import skimage.transform
import csv
import collections
import os
import time
from multiprocessing import Process , Manager

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True).astype(np.uint8)
            frames.append(frame)
        else:
            continue

    return np.array(frames).astype(np.uint8)


def getVideoList(data_path):
    '''
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    '''
    result = {}

    with open (data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od



def load_data(mode,down_factor=12 , th_count = 4,video_root="data/TrimmedVideos/"):

    def load_label(mode="train"):
        p = os.path.join( video_root , "label/gt_{}.csv".format(mode) )
        data = np.genfromtxt(p , dtype="str")
        label = {}
        keys = data[0].split(",")[0:6]
        for k in keys:
            label[k]=[]
        for r in data[1:]:
            tmp = r.split(",")[0:6]
            for i,k in enumerate(keys):
                label[k].append(tmp[i])
        return label

    gt_train_dict = load_label(mode=mode)

    gt_train_dict.keys()

    tmp_video_idx = gt_train_dict["Video_index"]
    tmp_video_cate = gt_train_dict["Video_category"]
    tmp_video_name = gt_train_dict["Video_name"]
    tmp_video_Action_labels = np.array(gt_train_dict["Action_labels"]).astype("int")

    
    manager = Manager()
    shared_img_arr = manager.list([i for i in range(th_count)])
    shared_label_arr = manager.list([i for i in range(th_count)])
    shared_length_arr = manager.list([i for i in range(th_count)])
    
    # shared_img_arr = [i for i in range(th_count)]
    # shared_label_arr = [i for i in range(th_count)]

    def multi_load(my_id , shared_img_array , shared_label_array , shared_length_array , mode="train"):
        print("#{} loading process".format(my_id))
        start_time = time.time()

        count = len(tmp_video_idx)//th_count
        start = my_id*count
        train_image = []
        train_label = []
        train_length = []
        
    #     end = start+count//100
        if my_id == th_count-1:
            end = len(tmp_video_idx)
        else:
            end = start+count
        c = 0
        for idx in tmp_video_idx[start:end]:
            c+=1
            if c == (count//2):
                print("#{} process finishes half of job. time : {:.2f}".format(my_id,time.time()-start_time))

            idx = int(idx)-1
            tmp_image = readShortVideo(os.path.join(video_root , "video/{}".format(mode)) , tmp_video_cate[idx] 
                                       , tmp_video_name[idx] , downsample_factor=down_factor)
            train_image.append(tmp_image)
            tmp_label = tmp_video_Action_labels[idx]
            l = len(tmp_image)
            train_label.extend([tmp_label]*l)
            train_length.append(l)

        shared_img_array[my_id] = np.concatenate(train_image , axis=0)
        shared_label_array[my_id] = np.array(train_label)
        shared_length_array[my_id] = np.array(train_length)
    
        print("#{} process finishes job. time : {:.2f}".format(my_id,time.time()-start_time))


    print("Multi-process : loading image...\n")
    start_time = time.time()
    job_list = []
    for i in range(th_count):
        config_dict = {"my_id":i ,  "shared_img_array":shared_img_arr 
                       ,"shared_label_array":shared_label_arr , "shared_length_array":shared_length_arr,"mode":mode}
        p = Process(target=multi_load , kwargs=config_dict)
        p.start()
        job_list.append(p)

    for i in range(th_count):
        job_list[i].join()

    print()
    print("Finish loading. Consume time : {}".format(time.time()-start_time))

    train_image = shared_img_arr[0]
    train_label = shared_label_arr[0]
    train_length = shared_length_arr[0]
    for i in range(1,th_count):
        train_image = np.concatenate([train_image , shared_img_arr[i]] , axis=0)
        train_label = np.concatenate([train_label , shared_label_arr[i]] , axis=-1)
        train_length = np.concatenate([train_length , shared_length_arr[i]] , axis=-1)
    print("Shape of {} image :".format(mode) , train_image.shape)
    print("Shape of {} label :".format(mode) , train_label.shape)
    print("Count of {} video :".format(mode) , train_length.shape)
    

    del tmp_video_idx, tmp_video_cate , tmp_video_name ,tmp_video_Action_labels
    return train_image , train_label , train_length
