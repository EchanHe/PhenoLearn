import json
import cv2
import glob 
import os
# import matplotlib.pyplot as plt

#dataloader
import torchvision.transforms as T
import torch 
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
import sys

import datetime
import time

from PIL import Image


# from detection.engine import train_one_epoch
# from detection.engine import evaluate
# from detection import utils

from deeplearning.detection.engine import train_one_epoch
from deeplearning.detection.engine import evaluate
from deeplearning.detection import utils


# def show_result(data,no_images=50):

    # figsize=(500, 1280)
    # columns = 1
    # rows = 50
    # fig=plt.figure()
    # j=0
    # for i,(image,keypoints) in enumerate(data):

    #     image=image.cpu().permute(1, 2, 0).numpy()
    #     image=image[:, :, [2, 1, 0]]  
    #     cX1=int(keypoints['keypoints'][0][0][0])
    #     cY1=int(keypoints['keypoints'][0][0][1])
    #     cX=int(keypoints['keypoints'][0][1][0])
    #     cY=int(keypoints['keypoints'][0][1][1])
    #     im = Image.fromarray(np.uint8((image)*255))
    #     image=np.asarray(im)
    #     cv2.line(image, (cX, cY), (cX1, cY1), (256,0,0), 10)
    #     cv2.circle(image, (cX, cY), 10, (255, 0, 0))
    #     cv2.circle(image, (cX1, cY1), 10, (0, 255, 255))
    #     plt.imshow(image)

    # plt.show()

def df_to_kpt_rcnn(df, idx:int,scale=1):
    """Dataframe from csv to keypoint in keypoint rcnn
    
    [N, K, 3]
    Args:
        df (_type_): dataframe
        idx (int): index of the data from dataset.__getitem__(idx).
        scale (int, optional): The scale . Defaults to 1.

    Returns:
        : list  [x,y, visibility] visibility 1: visible. 0: not visible
    """    
    pts=[]
    # extract the point names
    cols = df.columns
    col_names = ["_".join(col.split("_")[:-1]) for col in cols]
    col_names = pd.unique(col_names)
    #get the row using idx
    row = df.loc[df.index[idx],]

    #iterate through the name and use column <name>_x and <name>_y as the x and y
    for col_name in col_names:
        
        # x= row[col_name+"_x"]//scale
        # y = row[col_name+"_y"]//scale
        
        x= int(row[col_name+"_x"]//scale)
        y = int(row[col_name+"_y"]//scale)
        if not (x==-1 or y==-1 or np.isnan(x) or np.isnan(x)):
            pts+=[x,y,1]
        else:
            pts+=[0,0,0]
    return pts

def get_num_pts_from_df(df):
    """get the number of the keypoints using dataframe

    Args:
        df (_type_): dataframe

    Returns:
        _type_: number of the keypoints
    """
    cols = df.columns
    col_names = ["_".join(col.split("_")[:-1]) for col in cols]
    col_names = pd.unique(col_names)
    
    return len(col_names)


def out_to_csv(df, outs, scale=1):
    """Turn the keypoints from the output of kpt_rcnn to dataframe
    Used in predictions

    Args:
        df (_type_): A dataframe that has only a column of image names for the prediction
        outs (_type_): Output from kpt_rcnn, a list of dict
        scale (int, optional): The scale used for the keypoints and images, orignal x = scale * output x . Defaults to 1.

    Returns:
        _type_: dataframe with columns of x and y of the points.
    """    

    num_kpt =outs[0]['keypoints'].cpu().detach().numpy().shape[1]
    for id, out in enumerate(outs):
        keypoints = out["keypoints"].cpu().detach().numpy()
        
        for i_kpt in range(num_kpt):

            df.loc[df.index[id] , str(i_kpt)+"_x"] = int(keypoints[0][i_kpt][0] * scale)
            df.loc[df.index[id] , str(i_kpt)+"_y"] = int(keypoints[0][i_kpt][1] * scale)

    return df
    
def load_model_ckpt(model, ckpt):
    """load the checkpoint params to the model

    Args:
        model (_type_): DL model
        ckpt (_type_): path to the saved file.
    """
    try:
        model.load_state_dict(torch.load(ckpt))
    except Exception as e:
        print(e)
        
        
class kptDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, df, scale=1):
        """init function

        Args:
            image_path (_type_): Image folder
            df (_type_): dataframe of the image names and keypoints
            scale (int, optional): scale to resize the image and keypoint coordinates. 
                Width for training = original width//scale. x for training = original y//scale. Defaults to 1.
        """        
        # device, data type 
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.to_tensor = T.ToTensor()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.df = df
        self.image_path = image_path
        self.scale =scale
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
                
        image =Image.open(os.path.join(self.image_path, self.df.index[idx])).convert("RGB")

        w,h = image.size        
        image = image.resize((int(w//self.scale) , int(h//self.scale)))
        w,h = image.size

        # box is the same size as the image
        boxes=[[0,  0,w, h ]]
        boxes=np.array(boxes).astype(np.int16)

        # use df_to_kpt_rcnn() to convert df to [x,y,visibility]
        pts = df_to_kpt_rcnn(self.df, idx,self.scale)
        kpts=np.array(pts).astype(np.float32).reshape([1,-1,3])


        #all one for the label
        labels = np.ones((1), dtype=np.int8)

        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = torch.from_numpy(boxes).type(self.dtype).to(self.device)
        target["labels"] = torch.from_numpy(labels).type(torch.int64).to(self.device)
        target["keypoints"] = torch.from_numpy(kpts).type(self.dtype).to(self.device)
        target["image_id"] = image_id.to(self.device)

        img = self.to_tensor(image).to(self.device)        

        return img, target

class kptDataset_pred(torch.utils.data.Dataset):
    """Dataset object for predicting
    only return the image idx and images
    """
    def __init__(self, image_path, df, scale=1):
        
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.to_tensor = T.ToTensor()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.df = df
        self.image_path = image_path
        self.scale =scale
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        
        image =Image.open(os.path.join(self.image_path, self.df.index[idx])).convert("RGB")
        w,h = image.size
        image = image.resize((int(w//self.scale) , int(h//self.scale)))
        img = self.to_tensor(image).to(self.device)   
        return img
    
def get_model(num_kpts,fine_tune=True,train_fpn=True,train_kptHead=0, ):
    """Get the keypoint rcnn model with number of keypoint configured and how many params for training.

    Args:
        num_kpts (_type_): Number of key points.
        fine_tune (bool, optional): Whether fine tuning, i.e. training the whole network or not. Defaults to True.
        train_fpn (bool, optional): Whether train the feature pyramid network. Defaults to True.
        train_kptHead (int, optional): How many layers (0: no layers, 8 maximum layers) you want to train the keypoint head (layers of conv 2d). Defaults to 0.

    Returns:
        _type_: _description_
    """    
    #check gpu and set the device and data type
    is_available = torch.cuda.is_available()
    device =torch.device('cuda:0' if is_available else 'cpu')
    dtype = torch.cuda.FloatTensor if is_available else torch.FloatTensor
    
    # build model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False ,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_kpts)
    

    if fine_tune== True:
        # if fine tuning, set all params require grad (not for training) to false first
        for i,param in enumerate(model.parameters()):
            param.requires_grad = False

        # Set params for keypoint_head
        # parameters of a conv2d have two params, one weight, one bias
        if train_kptHead!=0:
            for i, param in enumerate(model.roi_heads.keypoint_head.parameters()):
                if i/2>=model.roi_heads.keypoint_head.__len__()/2-train_kptHead:
                    param.requires_grad = True

        # Set the params for the feature pyramid network
        if train_fpn==True:
            for param in model.backbone.fpn.parameters():
                param.requires_grad = True
    else:
        # if not fine tuning, train all the params
        for i,param in enumerate(model.parameters()):
            param.requires_grad = True

    # # set the keypoint predictor
    # out = nn.ConvTranspose2d(512, num_kpts, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # model.roi_heads.keypoint_predictor.kps_score_lowres = out
    
    for param in model.roi_heads.keypoint_predictor.parameters():
        param.requires_grad = True
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total: {}, trainable: {}".format(pytorch_total_params,pytorch_total_trainable_params))

    return model


def get_pred_model(num_kpts):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False)
    out = nn.ConvTranspose2d(512, num_kpts, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    model.roi_heads.keypoint_predictor.kps_score_lowres = out
    
    return model

def train(csv_path,img_path,scale, lr, batch,num_epochs, test_percent,train_lv,qt_signal):
    # initialize writer
    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Create an experiment name
    run_name = "point"
    # Use the date and time string and experiment name to create a log directory
    log_dir = f"runs/{run_name}_{time_date}"
    writer = SummaryWriter(log_dir)

    save_path = "saved_model/point_model_{}.pth".format(time_date)

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')    
    if qt_signal:
        qt_signal.emit("Training on: {}".format(device) )
    
    

    
    dir = os.path.dirname(os.path.abspath(save_path))

    if not os.path.exists(dir):os.makedirs(dir)
    
    df=pd.read_csv(csv_path, index_col='file')
    num_kpts = get_num_pts_from_df(df)
    
    Dataset = kptDataset(img_path, df, scale=scale)

    l=Dataset.__len__()
    
    torch.manual_seed(1)
    indices = torch.randperm(len(Dataset)).tolist()
    dataset = torch.utils.data.Subset(Dataset, indices[:-int(np.ceil(l*test_percent/100))])
    dataset_test = torch.utils.data.Subset(Dataset, indices[int(-np.ceil(l*test_percent/100)):])

    # define training and validation data loaders

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, shuffle=True,collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False,collate_fn=utils.collate_fn)    
    
    #The train_lv shows the level of the model what to been trained
    if train_lv == '1':      
        model=get_model(num_kpts = num_kpts ,train_fpn=False,train_kptHead=2,fine_tune=True)
    elif train_lv =='2':
        model=get_model(num_kpts = num_kpts ,train_fpn=True,train_kptHead=4,fine_tune=True)
    elif train_lv=='3':
        model=get_model(num_kpts = num_kpts ,train_fpn=False,train_kptHead=False,fine_tune=False)
        
    model=model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=2,
                                                gamma=0.9)

    start_time = time.time()
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        writer.add_scalar("Loss/train", metric_logger.meters['loss'].total, epoch)
        # update the learning rate
        lr_scheduler.step()
        

        metric_logger_valid , oks_values = evaluate(model, data_loader_test, device)
        writer.add_scalar("Loss/valid", metric_logger_valid.meters['loss'].value, epoch)    
        for i_oks , oks_value in enumerate(np.mean(oks_values,0)):
            writer.add_scalar("accuracy/{}".format(i_oks), oks_value, epoch)
        
        if qt_signal:
            qt_signal.emit(str(epoch)) 
            # qt_signal.pbar.setValue(epoch)


    end_time = time.time()

    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    running_time = "{} {} {}".format(hours, minutes, seconds)
    print("Running:",running_time)

    if qt_signal:    
        qt_signal.emit("Training complete\nTraining Time: {}\nThe model is saved: {}".format(running_time,save_path)) 
   
    torch.save(model, save_path)

def pred(csv_path,img_path, model_path,output_dir, scale,qt_signal):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    
    qt_signal.emit("Predicting on: {}".format(device) )
    
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    
    output_filename = "point_out_{}.csv".format(time)
    output_filename= os.path.abspath(os.path.join(output_dir,output_filename))
    
    df_pred = pd.read_csv(csv_path, index_col="file")
    dataset_pred = kptDataset_pred(img_path, df_pred, scale=scale)

    data_loader_pred = torch.utils.data.DataLoader(
        dataset_pred, batch_size=1, shuffle=False)
        
    
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    out_list=[]
    for idx, img in enumerate(data_loader_pred):
        img = img.to(device)

        out=model(img)
        out_list+=out
        qt_signal.emit(str(idx+1) )
    
    df_out = out_to_csv(df_pred, out_list,scale=scale)
    df_out.to_csv(output_filename)
    qt_signal.emit("Predicting finish, file is saved as:{}".format(output_filename) )

if __name__=="__main__":
    
    
    
    #arguments from inputs
    scale =10
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device=torch.device('cpu')
    csv_path = "data/pts.csv"
    img_path = "data/bird_10/"

    # save_path = "ckpt/test.pth"
    
    # time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # save_path = "saved_model/pt_{}.pth".format(time)
    
    test_percent=10
    num_epochs =20
    lr= 0.001
    batch_size = 2

    num_kpts = 3
    is_train=True

    train(csv_path,img_path,scale, lr, batch_size,num_epochs, test_percent,train_lv='1',qt_signal=False)

    # # for predicting
    # pred_csv_path = "data\\shell_pred\\df_img.csv"
    # pred_img_path ="data\\shell_pred\\"
    # ckpt_path = "ckpt\\kptrcnn.pth"
    # output_path="output/output.csv"

    # if is_train:
    #     ## data
    #     df=pd.read_csv(csv_path, index_col='file')

    #     Dataset = kptDataset(img_path, df, scale=scale)

    #     l=Dataset.__len__()
        
    #     torch.manual_seed(1)
    #     indices = torch.randperm(len(Dataset)).tolist()
    #     dataset = torch.utils.data.Subset(Dataset, indices[:-int(np.ceil(l*test_percent/100))])
    #     dataset_test = torch.utils.data.Subset(Dataset, indices[int(-np.ceil(l*test_percent/100)):])

    #     # define training and validation data loaders

    #     data_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True,collate_fn=utils.collate_fn)

    #     data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test, batch_size=1, shuffle=False,collate_fn=utils.collate_fn)


    #     ### data

    #     model=get_model(num_kpts = num_kpts,train_fpn=False,train_kptHead=False,fine_tune=False)

    #     model.to(device)
    #     params = [p for p in model.parameters() if p.requires_grad]
    #     optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                 step_size=3,
    #                                                 gamma=0.9)

        

    #     for epoch in range(num_epochs):
    #         # train for one epoch, printing every 10 iterations
    #         evaluate(model, data_loader, device)
            
            
    #         train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
            
    #         # update the learning rate
    #         lr_scheduler.step()
    #     evaluate(model, data_loader, device)
    #     images,targets = next(iter(data_loader))
    #     # images = list(image for image in images)
    #     # print(images.shape)
    #     # model.eval()
    #     # out=model(images)
        
    #     # show_result(zip(images, targets))
    #     # show_result(zip(images, out))
        
    #     torch.save(model.state_dict(), save_path)
    # else:    
    #     df_pred = pd.read_csv(pred_csv_path, index_col="file")
    #     dataset_pred = kptDataset_pred(pred_img_path, df_pred, scale=10)

    #     data_loader_pred = torch.utils.data.DataLoader(
    #         dataset_pred, batch_size=1, shuffle=False)
        
    #     model=get_pred_model(num_kpts = 2)
        
    #     load_model_ckpt(model,"/content/drive/My Drive/Colab Notebooks/ckpt/kptrcnn.pth" )
        
    #     model = get_pred_model(num_kpts)
    #     out_list=[]
    #     for img in data_loader_pred:
    #         out=model(img)
    #         out_list+=out
    #     df_out = out_to_csv(df_pred, out_list,scale=10)
    #     df_out.to_csv(output_path)