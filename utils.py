'''
@Author: Mingming Ren 任铭铭 
@Date: 2024-11-26 18:49:12 
@Last Modified by:   Mingming Ren ren 
@Last Modified time: 2025-07-06 18:49:12 
'''
import os, sys
import time
import logging
import pandas as pd
import numpy as np
import torch


def get_log_dir(root, prefix='', postfix=''):
    log_name = prefix + time.strftime('%y$m$d_%H%M%S', time.localtime()) + postfix
    log_dir = os.path.join(root, log_name, 'log')
    os.makedirs(log_dir)
    return log_dir, log_name

def creat_logger(log_name, log_dir=None, file_name='log'):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, file_name+'.txt'), mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info('Output and logs will be saved to: {}'.format(log_dir))
    return logger

def get_logger(args, log_dir, log_name, file_name, model=None):
    logger = creat_logger(log_name=log_name, log_dir=log_dir, file_name=file_name)
    logger.info('Command: {}'.format(' '.join(sys.argv)))
    arg_str = '\n'.join(['  {}: {}'.format(op, getattr(args, op)) for op in vars(args)])
    logger.info('Arguments:\n' + arg_str)
    if model is not None:
        logger.info(repr(model))
    return logger

def load_data(data_path, sep=' '):
    data = pd.read_table(data_path, header=None, sep=sep)
    data = np.array(data)
    points = data[:, :3]
    normals = data[:, 3:]
    return points, normals, len(points)

def normal_unitilize(normals):
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / norm


cnames = {
    'orangered': '#FF4500',
    'gold': '#FFD700',
    'green': '#008000',
    'hotpink': '#FF69B4',
    'deepskyblue': '#00BFFF',
    'dodgerblue': '#1E90FF',
    'firebrick': '#B22222',
    'forestgreen': '#228B22',
    'indianred': '#CD5C5C',
    'indigo': '#4B0082',
    'lemonchiffon': '#FFFACD',
    'lightseagreen': '#20B2AA',
    'aqua': '#00FFFF',
    'greenyellow': '#ADFF2F',
    'blue': '#0000FF',
    'blueviolet': '#8A2BE2',
    'brown': '#A52A2A',
    'cadetblue': '#5F9EA0',
    'chartreuse': '#7FFF00',
    'chocolate': '#D2691E',
    'coral': '#FF7F50',
    'crimson': '#DC143C',
    'darkcyan': '#008B8B',
    'darkgoldenrod': '#B8860B',
    'darkgreen': '#006400',
    'darkmagenta': '#8B008B',
    'darkorange': '#FF8C00',
    'darkorchid': '#9932CC',
    'darkred': '#8B0000',
    'darkviolet': '#9400D3',
    'lime': '#00FF00',
    'limegreen': '#32CD32',
    'magenta': '#FF00FF',
    'maroon': '#800000',
    'mediumorchid': '#BA55D3',
    'mediumseagreen': '#3CB371',
    'mediumspringgreen': '#00FA9A',
    'mediumturquoise': '#48D1CC',
    'mediumvioletred': '#C71585',
    'orange': '#FFA500',
    'orchid': '#DA70D6',
    'palevioletred': '#DB7093',
    'peachpuff': '#FFDAB9',
    'pink': '#FFC0CB',
    'plum': '#DDA0DD',
    'purple': '#800080',
    'red': '#FF0000',
    'rosybrown': '#BC8F8F',
    'royalblue': '#4169E1',
    'saddlebrown': '#8B4513',
    'salmon': '#FA8072',
    'sandybrown': '#FAA460',
    'seagreen': '#2E8B57',
    'slateblue': '#6A5ACD',
    'springgreen': '#00FF7F',
    'steelblue': '#4682B4',
    'tomato': '#FF6347',
    'turquoise': '#40E0D0',
    'violet': '#EE82EE',
    'yellow': '#FFFF00',
    'yellowgreen': '#9ACD32'
}
  
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def save_result(path, data):
    with open(path, 'w') as f:
        for elem in data:
            for item in elem:
                f.write(str(item[0].item()))
                f.write(',')
                f.write(str(item[1].item()))
                f.write(',')
                f.write(str(item[2].item()))
                f.write(',')
                f.write(str(item[3].item()))
                f.write(',')
                f.write(str(item[4].item()))
                f.write(',')
                f.write(str(item[5].item()))
                f.write(',')
                f.write(str(int(item[6].item())))
                f.write('\n')