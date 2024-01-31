import os
import numpy as np
import pandas as pd
import cv2
import re
import datetime

import torch
import torch.utils.data
from torchvision.io import read_image
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

# import matplotlib.pyplot as plt

""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import time


from deeplearning.segmentation.engine import train_one_epoch
from deeplearning.segmentation.engine import evaluate

# from segmentation.engine import train_one_epoch
# from segmentation.engine import evaluate


def contour_to_seg(seg,contour_coords,value, scale):
    """Turn the contour cell from df to a segmentation

    Args:
        seg (_type_): segmentation. shape (height, width)
        contour_coords (_type_): contours in opencv format [[[x11,y11],[x12,y12]] , [[x21,y21],[x22,y22]]]
        value (_type_): The value assign to the seg using this contour_coords
        scale: The scale used to resize the coordinates, e.g. x11=x11//scale 

    Returns:
        _type_: segmentation. shape (height, width)
    """
    contour_coords = eval(contour_coords)
    contour_cv = [(np.array(contour, dtype='int32')//scale).astype('int32') for contour in contour_coords]
    cv2.fillPoly(seg, contour_cv, value)
    return seg

def seg_to_mask(seg,n_cl):
    """Segmentation to mask

    Args:
        seg (_type_): segmentation. shape (height, width), the value can be 0 to (n_cl-1)
        n_cl (_type_): The number of classes for this segmentation

    Returns:
        _type_: mask, shape [height, width, n_cl]. value can only be 0,1
    """
    assert len(seg.shape) ==2 , "Make sure input is [height, width]"

    cl = np.unique(seg)
    h,w =seg.shape
    masks = np.zeros((h, w , n_cl))
    
    for i, c in enumerate(cl):
        masks[:, : , i] = seg == c

    masks = masks.astype('uint8')

    return masks


class segDataset(torch.utils.data.Dataset):
    """Dataset class for segmentation
    """    
    def __init__(self, img_path, mask_path=None, df_path=None, is_train=True,transforms=None,scale=1):
        """Init function

        Args:
            img_path (_type_): Image folder
            mask_path (_type_, optional): Mask folder. Defaults to None.
            df_path (_type_, optional): dataframe of the image names and segmentation. Defaults to None.
            train (bool, optional): _description_. Defaults to True.
            transforms (_type_, optional): _description_. Defaults to None.
        """        
        

        self.transforms = transforms
        self.scale = scale
        self.to_tensor = T.ToTensor()
        
        self.is_train=is_train
        # load all image files, sorting them to
        # ensure that they are aligned

        if mask_path!=None:
            self.img_path = img_path
            self.mask_path = mask_path

            self.imgs = list(sorted(os.listdir(img_path)))
            self.masks = list(sorted(os.listdir(mask_path)))

            self.mode_df =False
        elif df_path!=None:
            self.img_path = img_path
            self.df = pd.read_csv(df_path,index_col='file')

            self.imgs = self.df.index.values

            self.mode_df =True

    def __getitem__(self, idx):
        transform = T.Compose([T.ToTensor()])
        
        # load images
        img_name_path = os.path.join(self.img_path, self.imgs[idx])
        # PIL read and resize img (RGB)
        img = Image.open(img_name_path).convert("RGB")
        w,h = img.size   
        w_resized = int(w//self.scale)     
        h_resized = int(h//self.scale)     
        img = img.resize(( w_resized, h_resized))
        
        if not self.is_train:
            img_info={}
            img_info['w']=w
            img_info['h']=h
            
            return (self.to_tensor(img),img_info)
        
        if self.mode_df == False:
            mask_name_path = os.path.join(self.mask_path, self.masks[idx])
            # Read and resize Mask (grey value)
            mask = Image.open(mask_name_path).convert('L')
            mask = mask.resize((w_resized , h_resized))
            w_mask,h_mask = mask.size
            mask = np.array(mask)

            mask_temp = np.zeros((h_mask,w_mask,2))
            mask_temp[...,0] = np.where(mask == 255, mask_temp[...,0], 1)
            mask_temp[...,1] = np.where(mask != 255, mask_temp[...,1], 1)

            mask_tensor = self.to_tensor(mask_temp)
            # mask_tensor = torch.from_numpy(mask_temp.transpose(2, 0, 1))

        else: 
            row = self.df.loc[self.imgs[idx],:]
            colnames=self.df.columns
            
            seg = np.zeros((h_resized, w_resized))
            for i_col, col in enumerate(colnames):
                contour_coords = row[col]
                value=i_col+1
                seg = contour_to_seg(seg, contour_coords,value,self.scale)

            mask_temp = seg_to_mask(seg,n_cl=len(colnames)+1)
            
            mask_tensor = torch.from_numpy(mask_temp.transpose(2, 0, 1))
    
        # # instances are encoded as different colors
        # obj_ids = np.unique(mask)
        # # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # # split the color-encoded mask into a set
        # # of binary masks
        # masks = mask == obj_ids[:, None, None]


        img_tensor = self.to_tensor(img)

        # Customised transformation.
        # if self.transforms is not None:
        #     img, mask = self.transforms(img, mask)

        
        return img_tensor,mask_tensor

    def __len__(self):
        return len(self.imgs)



def get_model(outputchannels):
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    
    for i,param in enumerate(model.parameters()):
        param.requires_grad = False
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    # model.train()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total: {}, trainable: {}".format(pytorch_total_params,pytorch_total_trainable_params))
    return model
def get_model_with_lv(outputchannels, train_lv):
    """DeepLabv3 class with custom output channels and set the trainable params based on train_lv
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
        train_lv: the level of the training, 1 smallest, 3 largest
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    
    for i,params in enumerate(model.parameters()):
        params.requires_grad = False
        
    if train_lv==3:
        for params in model.backbone.layer3.parameters():
            params.requires_grad = True
    if train_lv>=2:
        for params in model.backbone.layer1.parameters():
            params.requires_grad = True
        for params in model.backbone.layer2.parameters():
            params.requires_grad = True
        for params in model.backbone.layer4.parameters():
            params.requires_grad = True    
            
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    # model.train()
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total: {}, trainable: {}".format(pytorch_total_params,pytorch_total_trainable_params))
    return model

def train(img_path,scale, lr, batch,num_epochs, test_percent,train_lv,qt_signal=False,csv_path=None,mask_path=None):
    # initialize writer
    time_date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Create an experiment name
    run_name = "segmentation"
    # Use the date and time string and experiment name to create a log directory
    log_dir = f"runs/{run_name}_{time_date}"
    writer = SummaryWriter(log_dir)

    
    save_path = os.path.abspath("saved_model/seg_model_{}.pth".format(time_date))
    
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if qt_signal:
        qt_signal.emit("Training on: {}".format(device) )
    
    img_path = os.path.abspath(img_path)
    
    

       
    dir = os.path.dirname(os.path.abspath(save_path))
    if not os.path.exists(dir):os.makedirs(dir)
    
    # init the dataset and number of class
    if csv_path!=None:
        csv_path = os.path.abspath(csv_path)
        df=pd.read_csv(csv_path, index_col='file')
        num_classes = len(df.columns)+1
    
        dataset = segDataset(img_path = img_path, df_path = csv_path,scale=scale)
    elif mask_path!=None:
        num_classes=2
        
        dataset = segDataset(img_path = img_path, mask_path = mask_path,scale=scale)
    
    # Split and create dataloader
    l=dataset.__len__()
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-int(np.ceil(l*test_percent/100))])
    dataset_test = torch.utils.data.Subset(dataset, indices[int(-np.ceil(l*test_percent/100)):])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)    

    model =get_model_with_lv(outputchannels=num_classes, train_lv=int(train_lv))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=2,
                                            gamma=0.9)
        # Train and evaluate model
        # Log the training detail
    start_time = time.time()
        
    for epoch in range(num_epochs):
        
        
        metric_logger = train_one_epoch(model, loss_fn,optimizer, data_loader,None, device, epoch, print_freq=1)
        writer.add_scalar("Loss/train", metric_logger.meters['loss'].total, epoch)
        scheduler.step()
        
        metric_logger_valid, confmat, dice_per_class = evaluate(model, data_loader_test, device, num_classes, loss_fn)
        writer.add_scalar("Loss/valid", metric_logger_valid.meters['loss'].total, epoch)
        
        # acc_global, accs, _ = confmat.compute()
        # writer.add_scalar("accuracy/ave", acc_global, epoch)
        # for i_ac, ac in enumerate(accs):
        #     writer.add_scalar("accuracy/{}".format(i_ac), ac, epoch)
        
        dice_per_class = torch.mean(dice_per_class,0).detach().cpu().numpy()
        writer.add_scalar("accuracy/ave", np.mean(dice_per_class), epoch)
        for i_dice,dice in enumerate(dice_per_class):
            writer.add_scalar("accuracy/{}".format(i_dice), dice, epoch)
        
        if qt_signal:
            qt_signal.emit(str(epoch)) 
            
    end_time = time.time()

    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    running_time = "{} {} {}".format(hours, minutes, seconds)
    print("Running:",running_time)        
    
    if qt_signal:    
        qt_signal.emit("Training complete\nTraining Time: {}\nThe model is saved: {}".format(running_time,save_path)) 
        
    torch.save(model, save_path)

def pred(csv_path,img_path, model_path,output_dir,format, scale,qt_signal):
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
    df=pd.read_csv(csv_path,index_col='file')
    
    qt_signal.emit("Predicting on: {}".format(device) )
    
    if format=="CSV":     
        time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        output_filename = "seg_output_{}.csv".format(time)
        output_filename= os.path.abspath(os.path.join(output_dir,output_filename))
    else:
        output_dir = os.path.join(output_dir,"mask")
        
    if not os.path.exists(output_dir):os.makedirs(output_dir)
    
    dataset_pred = segDataset(img_path = img_path, df_path = csv_path,scale=scale,is_train=False)

    data_loader_pred = torch.utils.data.DataLoader(
            dataset_pred, batch_size=1, shuffle=False)
    model = torch.load(model_path)
    model = model.to(device)
    model.eval()
    
    for idx, (img,img_info) in enumerate(data_loader_pred):
        img_name = df.index.values[idx]
         
        print("img info++++++++++",img_info)
        img = img.to(device)  
        out = model(img)
        
        out_temp = out['out'][0].cpu().detach().numpy()
        seg= out_temp.transpose(1, 2, 0).argmax(2)
        
        if format=="CSV":
            for idx_seg_class in range(1,11):
                if (seg==idx_seg_class).any():
                    output_mask = np.zeros(seg.shape).astype('uint8')
                    output_mask[seg==idx_seg_class]=255
                    
                    contours, _ = cv2.findContours(output_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    contours_str = str([(contour*scale).tolist() for contour in contours])
                    
                    if contours_str!="[]":
                        df.loc[img_name , str(idx_seg_class)]=contours_str
        else:
            output_mask = np.zeros(seg.shape).astype('uint8')
            output_mask[seg==1]=255
            if scale!=1:
                output_mask = cv2.resize(output_mask, (img_info['w'].item(),img_info['h'].item()),
                       interpolation = cv2.INTER_NEAREST )
            cv2.imwrite( os.path.join(output_dir,img_name) , output_mask)
        qt_signal.emit(str(idx+1) )
     
    if format=="CSV":   
        df.to_csv(output_filename)     
        qt_signal.emit("Predicting complete, output file is saved as:{}".format(output_filename) )
    else:
        qt_signal.emit("Predicting complete, Masks saved in:{} folder".format(output_dir) )

if __name__=="__main__":
    train("data/bird_10/",10, 0.001, 2,4, 20,1,csv_path='data/seg.csv',mask_path=None)      
    
    # device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # num_classes = 2
    # img_path = "data/test_10/img/"
    # mask_path = "data/test_10/mask_pl/"
    # batch = 2

    # num_epochs=1
    # test_percent=20
    # time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    # save_path = "saved_model/seg_{}.pth".format(time)

    # is_train=True

    # if is_train:
    #     # set the datasets dataloader
    #     dataset = segDataset(
    #         img_path = img_path, 
    #         mask_path = mask_path,
    #         scale=20
    #     )

        
    #     l=dataset.__len__()
    #     torch.manual_seed(1)
    #     indices = torch.randperm(len(dataset)).tolist()
    #     dataset = torch.utils.data.Subset(dataset, indices[:-int(np.ceil(l*test_percent/100))])
    #     dataset_test = torch.utils.data.Subset(dataset, indices[int(-np.ceil(l*test_percent/100)):])

    #     data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    #     data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False)


    #     model = get_model(outputchannels=num_classes)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    #     loss_fn = torch.nn.MSELoss()

    #     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        
    #     # Train and evaluate model
    #     # Log the training detail
    #     for epoch in range(num_epochs):
    #         train_one_epoch(model, loss_fn,optimizer, data_loader,scheduler, device, epoch, print_freq=1)
    #         evaluate(model, data_loader, device, num_classes)
        

    #     torch.save(model, save_path)

        # img = images[0].squeeze()
        # mask = masks[0].squeeze()

        # print(img.shape, img[0].shape, img[0].squeeze().shape)
        # print(mask.shape, mask[0].shape, mask[0].squeeze().shape)