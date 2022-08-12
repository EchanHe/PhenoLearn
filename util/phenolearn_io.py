import numpy as np
import pandas as pd
import cv2
import os
import json


class annotationError(Exception):
    pass

def transfer_df_to_json(df, pt_only, seg_only):
    id_col = 'file'
    if pt_only:
        coord_cols = list(df.columns)
        coord_cols.remove(id_col)
        coord_cols = [name[:-2] for name in coord_cols[0::2]]
        outline_cols = []
    elif seg_only:
        outline_cols = list(df.columns)
        outline_cols.remove(id_col)
        coord_cols = []
    data = []
    for idx, row in df.iterrows():
        temp = dict()
        temp['file_name'] = row[id_col]
        temp['points'] = []
        temp['outlines_cv'] = {}
        for col in coord_cols:
            # POint detail
            point = dict()
            point['name'] = col
            point['x'] = row[col+'_x']
            point['y'] = row[col+'_y']
            if point['x'] ==-1 or point['y'] == -1:
                point['absence'] = True
            else:
                point['absence'] = False
            temp['points'].append(point)

        for col in outline_cols:
            # POint detail
            temp['outlines_cv'][col] ={}
            temp['outlines_cv'][col]["contours"] = eval(row[col])


        data.append(temp)
    return data


def transfer_df_to_json_by_cols(df, id_col = 'file',coord_cols=[], outline_cols=[], prop_cols=[]):

    data = []
    # coord_cols= [name[:-2] for name in coord_cols[0::2]]

    coord_cols = list(set([coord[:-2] for coord in coord_cols]))

    for idx, row in df.iterrows():
        temp = dict()
        temp['file_name'] = row[id_col]
        temp['points'] = []
        temp['outlines_cv'] = {}
        temp['property'] = {}
        for col in coord_cols:
            # POint detail
            if (not np.isnan(row[col+'_x'])) and (not np.isnan(row[col+'_y'])) :
                point = dict()
                point['name'] = col
                point['x'] = row[col+'_x']
                point['y'] = row[col+'_y']
                if point['x'] ==-1 or point['y'] == -1:
                    point['absence'] = True
                else:
                    point['absence'] = False
                temp['points'].append(point)

        for col in outline_cols:
            # POint detail
            temp['outlines_cv'][col] ={}
            try:
                temp['outlines_cv'][col]["contours"] = eval(row[col])
            except:
                temp['outlines_cv'][col]["contours"] = []

        for col in prop_cols:
            temp['property'][col] = row[col]

        data.append(temp)
    return data

def add_to_dict(df_dict,key,value):
    if key in df_dict:
        df_dict[key].append(value)
    else:
        df_dict[key] = [value]

def transfer_json_to_df(data, pt_only, seg_only):
    df_dict={}
    df = pd.DataFrame()
    for entry in data:
        file_name = entry['file_name']

        points = entry.get('points', None)
        segs = entry.get("outlines_cv" , None)

        if points and pt_only:
            for pt in points:
                df.loc[file_name,pt['name']+'_x'] = pt['x']
                df.loc[file_name,pt['name']+'_y'] = pt['y']

        if segs and seg_only:
            for key, seg in segs.items():
                df.loc[file_name,key] = str(seg['contours'])


    # df = pd.DataFrame(df_dict)
    df.index.name = 'file'
    df = df.reset_index()

    return df
def write_seg_result(df_valid, pred_contours,path, data_cols, write_json=False):
    df_file_names = df_valid.drop(data_cols , axis=1 , errors = 'ignore')
    df_file_names = df_file_names.reset_index(drop=True)

    result = pd.DataFrame(pred_contours, columns = data_cols )
    result = pd.concat([df_file_names,result],axis=1)
    result.to_csv(path, index = False)

    if write_json:
        data = transfer_df_to_json(result, pt_only=False, seg_only= True)
        file_name = os.path.splitext(path)[0] + ".json"
        with open(file_name, "w") as write_file:
            json.dump(data, write_file)
    return result

def write_seg_performance(df_valid, iou, precision, recall, path, data_cols):
    # The first value is the metric of background
    data_cols = ["background"] + data_cols

    df_file_names = df_valid.drop(data_cols , axis=1 , errors = 'ignore')
    df_file_names = df_file_names.reset_index(drop=True)

    iou_cols = ["IOU_"+col for col in data_cols]
    df_iou = pd.DataFrame(iou, columns = iou_cols)

    precision_cols = ["Precision"+col for col in data_cols]
    df_precision = pd.DataFrame(precision, columns = precision_cols )

    recall_cols = ["Recall"+col for col in data_cols]
    df_recall = pd.DataFrame(recall, columns = recall_cols )


    result = pd.concat([df_file_names,df_iou, df_precision, df_recall],axis=1)
    result.to_csv(path, index = False)

    return result


def heatmap_to_coord(heatmaps , ori_width , ori_height):
    """
    # Goals: transfer heatmaps to the size given
    Params:
        heatmaps: heatmaps from pose esitmation method,
            Shape(df_size, heatmap_height,heatmap_width, heatmap channel)
        ori_width: The width you want to transfer back
        ori_height: The height you want to transfer back
    return:
        The coordinates found on heatmap, eg [x1,y1,x2,y2,...,xn,yn]
        Shape: (df_size,cnt_size*2)
    """
    df_size = heatmaps.shape[0]
    cnt_size = heatmaps.shape[3]
    output_result = np.ones((df_size,cnt_size*2))
    for i in range(df_size):
        for j in range(cnt_size):
            heat_map = heatmaps[i,:,:,j]
            ori_heatmap = cv2.resize(heat_map, dsize=(ori_width, ori_height),interpolation = cv2.INTER_NEAREST)

            # map_shape = np.unravel_index(np.argmax(ori_heatmap, axis=None), ori_heatmap.shape)
            # output_result[i,j*2+0] = map_shape[1] + 1
            # output_result[i,j*2+1] = map_shape[0] + 1


            y_x_coords = np.where(ori_heatmap == np.max(ori_heatmap))
            x = int(round(np.mean(y_x_coords[1]+1)))
            y = int(round(np.mean(y_x_coords[0]+1)))

            output_result[i,j*2+0] = x
            output_result[i,j*2+1] = y

    return output_result

def write_pt_result(df_valid,pred_coord, path, data_cols, write_json=False):
    df_file_names = df_valid.drop(data_cols , axis=1 , errors = 'ignore')
    df_file_names = df_file_names.reset_index(drop=True)

    result = pd.DataFrame(pred_coord, columns = data_cols )

    result = pd.concat([df_file_names,result],axis=1)

    result.to_csv(path, index = False)

    if write_json:
        data = transfer_df_to_json(result, pt_only=True, seg_only= False)
        file_name = os.path.splitext(path)[0] + ".json"
        with open(file_name, "w") as write_file:
            json.dump(data, write_file)


    return result

def write_pt_performance(df_valid, pixel_distance, path, data_cols, name_cols):
    """
    Write performance file
    :return:
    """

    df_file_names = df_valid.drop(data_cols , axis=1 , errors = 'ignore')
    df_file_names = df_file_names.reset_index(drop=True)
    result = pd.DataFrame(pixel_distance, columns = name_cols )

    result = pd.concat([df_file_names,result],axis=1)

    result.to_csv(path, index = False)
    return result



def check_df_format(df, type, istrain):
    print("check format")

    if 'file' not in df.columns:
        raise KeyError("There should be a file name")
    data_cols = list(df.columns)
    data_cols.remove('file')

    if istrain:
        if type == 'point':
            if len(data_cols)<2:
                raise ValueError("Coordinate columns are not enough")

            for col_x, col_y in zip(data_cols[::2], data_cols[1::2]):
                cond_1 = col_x[:-2] == col_y[:-2]
                cond_2 = (col_x[-2:] =='_x') and (col_y[-2:] =='_y')
                if not (cond_1 and cond_2):
                    raise ValueError("Check if coordinate columns are pt1_x, pt1_y, pt2_x, pt2_y")




def match_imgs_and_anno_file(img_dir, df):
    """
    Match files in the image directory and the annotation file
    :return:
    """

    files = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

    index = df.file.isin(files)
    if len(df.loc[index,:])==0:
        raise ValueError("No matched images are found in the directory")
    return df.loc[index,:]

def check_imgs(img_dir, df):
    """
    Check if images are the same resolution
    :param img_dir:
    :param df:
    :return:
    """
    resos = []
    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(img_dir, row['file']))
        resos.append("{}x{}".format(img.shape[1], img.shape[0]))

    if len(np.unique(resos))==1:
        return True
    else:
        raise ValueError("Images do not have the same resolution")

# def test_except(df,type,img_dir):
#     check_df_format(df, type)
#     check_imgs(img_dir, df)


def transfer_json_to_df_by_cols(data, coord_cols =False, outline_cols=False, prop_cols=False , mode='all'):
    """

    :param df:
    :param coord_cols: False: do not convert point data
    :param outline_cols: False: do not convert segmentation data
    :param prop_cols: False do not convert segmentation
    :return:
    """

    df = pd.DataFrame()
    for id_row, entry in enumerate(data):
        file_name = entry['file_name']

        points = entry.get('points', None)
        segs = entry.get("outlines_cv" , None)
        props = entry.get("property")

        df.loc[id_row , 'file'] =  file_name

        if mode=="all" or mode =="point":

            if points and (coord_cols != False):
                for idx, pt in enumerate(points):
                    if coord_cols is None:
                        df.loc[id_row, pt['name']+'_x'] = pt['x']
                        df.loc[id_row, pt['name']+'_y'] = pt['y']
                    else:
                        df.loc[id_row, coord_cols[idx]+'_x'] = pt['x']
                        df.loc[id_row, coord_cols[idx]+'_y'] = pt['y']

        if mode=="all" or mode =="seg":
            if segs and (outline_cols != False):
                for idx, (key, seg) in enumerate(segs.items()):
                    if outline_cols is None:
                        df.loc[id_row, key] = str(seg['contours'])
                    else:
                        df.loc[id_row,outline_cols[idx]] = str(seg['contours'])

        if mode=="all":
            if props and (prop_cols != False):
                for idx, (key, prop) in enumerate(props.items()):
                    if prop_cols is None:
                        df.loc[id_row, "prop_" + key] = str(prop)
                    else:
                        df.loc[id_row, prop_cols[idx]] = str(prop)
    return df

if __name__ == "__main__":    # train(anno_file="input_dir/pt.csv",img_dir="input_dir/",
    #       output_dir="output_dl/" , scale=25,lr=0.0001, batch=1,epochs=1,type='point',
    #       training_split=0.8)

    # match_imgs_and_anno_file(img_dir="./" , df = pd.read_csv("input_dir/pt.csv"))
    # try:
    #     # print(check_df_format( df = pd.read_csv("input_dir/seg.csv") , type= 'point'))
    #     # check_imgs(img_dir="input_dir/" , df = pd.read_csv("input_dir/pt.csv"))
    #
    #     test_except(df = pd.read_csv("input_dir/pt.csv"),type= 'point',img_dir="input_dir/")
    #
    # except KeyError as e:
    #     print("key",e.args)
    # except ValueError as e:
    #     print(e)

    ## Test csv conversion in
    with open("../data/shell_10/continue_meta.json", "r") as read_file:
        data = json.load(read_file)
        df = transfer_json_to_df_by_cols(data, coord_cols =None, outline_cols=None, prop_cols=None)

    df.to_csv("../data/shell_10/test.csv" , index = False)

    # with open("shell_10/continue.json", "r") as read_file:
    #     data = json.load(read_file)
    #
    # df_all =transfer_json_to_df(data, pt_only=False, seg_only=True)
    # print(df_all)
