'''
Author: airscker
Date: 2023-07-26 18:55:28
LastEditors: airscker
LastEditTime: 2023-07-26 20:53:39
Description: Visualization tools

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

def ShowDGLGraph(graph, nodeLabel: str = '', EdgeLabel: str = '', show: bool = True, save_path: str = '',figsize:tuple=(5,5)):
    """
    ## Visualizes a DGLGraph using networkx and matplotlib.

    ### Args:
        - graph (dgl.DGLGraph): The input DGLGraph object.
        - nodeLabel (str): The attribute name for node labels, if no feature exists, just give it ''.
        - EdgeLabel (str): The attribute name for edge labels, if no feature exists, just give it ''.
        - show (bool, optional): Whether to display the graph. Defaults to True.
        - save_path (str, optional): The path to save the graph image. Defaults to ''.

    ### Returns:
        int: 1 if successful, otherwise 0.
    """
    try:
        import networkx
    except ImportError:
        warnings.warn(f"Python module 'networkx' is not available, please install it to enable the visualization of graph data.")
        return 0
    try:
        import dgl
    except ImportError:
        warnings.warn("Python module 'dgl' is not available, please install it to enable the visualization of graph data.")
        return 0

    plt.figure(figsize=figsize)
    networkx_graph = graph.to_networkx(node_attrs=nodeLabel.split(), edge_attrs=EdgeLabel.split())
    nodes_pos = networkx.spring_layout(networkx_graph)
    networkx.draw(networkx_graph, nodes_pos, edge_color="grey", node_size=500, with_labels=True)

    node_data = networkx.get_node_attributes(networkx_graph, nodeLabel)
    node_labels = {index: "N:" + str(data) for index, data in enumerate(node_data)}

    pos_higher = {}
    for k, v in nodes_pos.items():
        if v[1] > 0:
            pos_higher[k] = (v[0] - 0.04, v[1] + 0.04)
        else:
            pos_higher[k] = (v[0] - 0.04, v[1] - 0.04)

    networkx.draw_networkx_labels(networkx_graph, pos_higher, labels=node_labels, font_color="brown", font_size=12)

    edge_labels = networkx.get_edge_attributes(networkx_graph, EdgeLabel)
    edge_labels = {(key[0], key[1]): "w:" + str(edge_labels[key].item()) for key in edge_labels}

    networkx.draw_networkx_edges(networkx_graph, nodes_pos, alpha=0.5)
    networkx.draw_networkx_edge_labels(networkx_graph, nodes_pos, edge_labels=edge_labels, font_size=12)

    if save_path != '':
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return 1

def plot_3d(img, save='', show=False, title='', norm=False,vector=None):
    """
    ## Plot the 3D image of the given image.

    ### Args:
        - img: the image data to plot
        - save: the path to save the image
        - show: whether to show the image
        - title: the title of the image
        - norm: whether to normalize the image
        - vector: (optional)the vector to be ploted, [x0,y0,z0,nx,ny,nz]
    """
    x = []
    y = []
    z = []
    num = []
    img = np.array(img)
    if norm:
        img = (img-np.min(img))/(np.max(img)-np.min(img))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if img[i][j][k] != 0:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    num.append(img[i][j][k])
    fig = plt.figure(figsize=(15, 15))
    plt.title(title)
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=num, cmap='jet')
    if vector is not None:
        start_point=vector[:3]
        direction=vector[3:]
        direction=direction/np.sqrt(np.sum(direction**2))
        end_point=direction*5+start_point
        ax.quiver(start_point[0],start_point[1],start_point[2],end_point[0],end_point[1],end_point[2],color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if save != '':
        plt.savefig(save, dpi=600)
    if show:
        plt.show()
    plt.clf()
    return 0


def plot_hist_2nd(data, title='x', bins=15, sigma=3, save='', show=False):
    """
    ## Plots a histogram of the data provided.
    It also includes lines to represent the mean and +/- 3 standard deviations.

    ### Args:
        - data: The list of numbers that will be plotted as a histogram
        - title: The title for the plot
        - bins: The number of bins to use for the histogram (default is 15).
        - sigma: How many standard deviations away from mean should be highlighted (default is 3).
        - save: The path to save the ploted image, if '' given, saving action will be canceled
        - show: Whether to show the ploted image within console
    """
    plt.figure(figsize=(20, 8))
    plt.title(f'Distribution of {title} Total Number: {len(data)}\
        \nMIN/MAX: {np.min(data)}/{np.max(data)} MEAN/STD: {np.mean(data)}/{np.std(data)}\
        \n{sigma}Sigma: {np.mean(data)+sigma*np.std(data)} {np.mean(data)-sigma*np.std(data)}')
    # n,bins,patchs=plt.hist(data,bins=list(np.arange(np.min(data)-0.01,np.max(data)+0.01,0.01)),rwidth=0.9)
    n, bins, patchs = plt.hist(data, bins=bins, rwidth=0.9)
    for i in range(len(n)):
        plt.text(bins[i], n[i]*1.02, round(n[i], 6),
                 fontsize=12, horizontalalignment="left")
    sigma_rangex = np.array(
        (np.mean(data)-sigma*np.std(data), np.mean(data)+sigma*np.std(data)))
    axis = plt.axis()
    plt.text(x=np.mean(sigma_rangex),
             y=axis[-1]/1.2, s=f'+/- {sigma} Sigma Range', ha='center')
    plt.text(x=np.mean([np.min(data), sigma_rangex[0]]), y=axis[-1]/2,
             s=f'Sample Number: {np.count_nonzero(data<sigma_rangex[0])}', ha='center')
    plt.text(x=np.mean([sigma_rangex[1], np.max(data)]), y=axis[-1]/2,
             s=f'Sample Number: {np.count_nonzero(data>sigma_rangex[1])}', ha='center')
    plt.fill_betweenx(y=axis[-2:], x1=max(np.min(data), sigma_rangex[0]),
                      x2=min(np.max(data), sigma_rangex[1]), alpha=0.2)
    plt.fill_betweenx(y=np.array(
        axis[-2:])/2, x1=np.min(data), x2=sigma_rangex[0], alpha=0.2)
    plt.fill_betweenx(y=np.array(
        axis[-2:])/2, x1=sigma_rangex[1], x2=np.max(data), alpha=0.2)
    plt.axis(axis)
    if save != '':
        plt.savefig(save)
    if show == True:
        plt.show()
    plt.clf()


def plot_curve(data, title='Curve', axis_label=['Epoch', 'Loss'], data_label=['Curve1'], save='', mod='min', show=False):
    """
    ## Plots a single or multiple curves on the same plot.
    The function takes in a list of data and labels for each curve to be plotted.
    The axis labels are optional, but if provided they will be used as x-axis label and y-axis label respectively.
    If no axis labels are provided, then the default values `Epoch` and `Loss&quot` will be used instead.

    ### Args:
        - data: Plot the data, it can be a list of numpy array or a single numpy array
        - title: Set the title of the plot
        - axis_label: Set the labels of the every axis
        - data_label: Labels of the curves
        - save: The path to save the ploted image, if '' given, saving action will be canceled
        - mod: plot the max/min value
        - show: Whether to show the ploted image within console
    """
    # data=np.array(data)
    plt.figure(figsize=(20, 10))
    plt.title(f'{title}')
    if isinstance(data[0], list) or isinstance(data[0], np.ndarray):
        for i in range(len(data)):
            data[i] = np.array(data[i])
            label = data_label[i] if len(data_label) >= i+1 else f'Curve{i+1}'
            if mod == 'min':
                label = f'{label} MIN/POS: {np.min(data[i])}/{np.argwhere(data[i]==np.min(data[i]))[-1]}'
                plt.axhline(np.min(data[i]), linestyle='-.')
            elif mod == 'max':
                label = f'{label} MAX/POS: {np.max(data[i])}/{np.argwhere(data[i]==np.max(data[i]))[-1]}'
                plt.axhline(np.max(data[i]), linestyle='-.')
            else:
                label = f'{label}\nMAX/POS: {np.max(data[i])}/{np.argwhere(data[i]==np.max(data[i]))[-1]}\nMIN/POS: {np.min(data[i])}/{np.argwhere(data[i]==np.min(data[i]))[-1]}'
            if isinstance(data[i][0],tuple) or isinstance(data[i][0],list) or isinstance(data[i][0],np.ndarray):
                plt.plot(*data[i], label=label)
            else:
                plt.plot(data[i],label=label)
    else:
        data = np.array(data)
        label = data_label[0] if len(data_label) >= 1 else f'Curve1'
        if mod == 'min':
            label = f'{label} MIN/POS: {np.min(data)}/{np.argwhere(data==np.min(data))[-1]}'
            plt.axhline(np.min(data), linestyle='-.')
        elif mod == 'max':
            label = f'{label} MAX/POS: {np.max(data)}/{np.argwhere(data==np.max(data))[-1]}'
            plt.axhline(np.max(data), linestyle='-.')
        else:
            label = f'{label}\nMAX/POS: {np.max(data)}/{np.argwhere(data==np.max(data))[-1]}\nMIN/POS: {np.min(data)}/{np.argwhere(data==np.min(data))[-1]}'
        if isinstance(data[0],tuple) or isinstance(data[0],list) or isinstance(data[0],np.ndarray):
            plt.plot(*data, label=label)
        else:
            plt.plot(data,label=label)
    plt.xlabel(axis_label[0])
    plt.ylabel(axis_label[1])
    plt.grid()
    plt.legend()
    if save != '':
        plt.savefig(save, dpi=400)
    if show:
        plt.show()
    plt.clf()
