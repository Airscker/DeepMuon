'''
Author: airscker
Date: 2023-01-30 18:27:14
LastEditors: airscker
LastEditTime: 2023-02-02 20:23:13
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''

import os
import math
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


torch.set_default_tensor_type(torch.DoubleTensor)
densenet_40_12_bc_weights_path = os.path.join(
    os.path.dirname(__file__), "pretrained_densenet_4012BC.pth.tar")


class RNN(nn.Module):

    def __init__(self, n_classes, input_size, hidden_size, rnn_type="LSTM", dropout=0.0, max_seq_len=15, attention=True, bidirectional=True, use_cuda=False):
        """
        Initalize RNN module

        :param n_classes:
        :param input_size:
        :param hidden_size:
        :param rnn_type:    GRU or LSTM
        :param dropout:
        :param max_seq_len:
        :param attention:
        :param bidirectional:
        :param use_cuda:
        """
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_classes = n_classes
        self.attention = attention
        self.max_seq_len = max_seq_len
        self.use_cuda = use_cuda

        self.rnn_type = rnn_type
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                input_size, hidden_size, batch_first=True, dropout=dropout, bidirectional=self.bidirectional)
        self.dropout = nn.Dropout(dropout)

        b = 2 if self.bidirectional else 1

        if attention:
            self.attn_linear_w_1 = nn.Linear(
                b * hidden_size, b * hidden_size, bias=True)
            self.attn_linear_w_1a = nn.Linear(
                b * hidden_size, b * hidden_size, bias=True)
            self.attn_linear_w_2 = nn.Linear(b * hidden_size, 1, bias=False)
            self.attn_linear_w_2a = nn.Linear(b * hidden_size, 1, bias=False)
            self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.linear = nn.Linear(b * hidden_size, n_classes)

    def embedding(self, x, hidden, x_mask=None):
        """
        Get learned representation
        """
        x_mask = self._get_mask(x) if not x_mask else x_mask

        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)

        if self.attention:
            output = self._two_fold_attn_pooling(output)
        else:
            output = self._mean_pooling(output, x_mask)

        return output

    def _mean_pooling(self, x, x_mask):
        """
        Mean pooling of RNN hidden states
        TODO: add variable sequence lengths back in
        """
        return torch.mean(x, 1)
        # x_lens = x_mask.data.eq(0).long().sum(dim=1)
        # if self.use_cuda:
        #     weights = Variable(torch.ones(x.size()).cuda() / x_lens.unsqueeze(1).float())
        # else:
        #     weights = Variable(torch.ones(x.size()) / x_lens.unsqueeze(1).float())
        # weights.data.masked_fill_(x_mask.data, 0.0)
        # output = torch.bmm(x.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        # return output

    def _attn_mean_pooling(self, x, x_mask):
        """
        Weighted mean pooling of RNN hidden states, where weights are
        calculated via an attention layer where the attention weight is
            a = T' . tanh(Wx + b)
            where x is the input, b is the bias.
        """
        emb_squish = torch.tanh(self.attn_linear_w_1(x))
        emb_attn = self.attn_linear_w_2(emb_squish)
        emb_attn.data.masked_fill_(x_mask.unsqueeze(2).data, float("-inf"))
        emb_attn_norm = torch.softmax(emb_attn.squeeze(2), dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(
            1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors

    def _two_fold_attn_pooling(self, x):
        emb_squish_1 = torch.tanh(self.attn_linear_w_1(x))
        emb_attn_1 = self.attn_linear_w_2(emb_squish_1)
        emb_squish_2 = torch.tanh(self.attn_linear_w_1a(x))
        emb_attn_2 = self.attn_linear_w_2a(emb_squish_2)
        alpha_limit = torch.sigmoid(self.alpha)
        emb_attn = alpha_limit * emb_attn_1 + (1 - alpha_limit) * emb_attn_2
        emb_attn_norm = torch.softmax(emb_attn.squeeze(2), dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(
            1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors

    def _get_mask(self, x):
        """
        Return an empty mask
        :param x:
        :return:
        """
        x_mask = Variable(torch.zeros(x.size(0), self.max_seq_len).byte())
        return x_mask.cuda() if self.use_cuda else x_mask

    def forward(self, x, hidden, x_mask=None):
        """
        Forward pass of the network

        :param x:
        :param hidden:
        :param x_mask: 0-1 byte mask for variable length sequences
        :return:
        """
        x_mask = self._get_mask(x) if not x_mask else x_mask

        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)

        if self.attention:
            output = self._two_fold_attn_pooling(output)
        else:
            output = self._mean_pooling(output, x_mask)

        output = self.linear(output)
        return output

    def init_hidden(self, batch_size):
        """
        Initialize hidden state params

        :param batch_size:
        :return:
        """
        b = 2 if self.bidirectional else 1
        if self.rnn_type == "LSTM":
            h0 = (Variable(torch.zeros(b, batch_size, self.hidden_size)),
                  Variable(torch.zeros(b, batch_size, self.hidden_size)))
            h0 = h0 if not self.use_cuda else [h0[0].cuda(), h0[1].cuda()]
        else:
            h0 = Variable(torch.zeros(b, batch_size, self.hidden_size))
            h0 = h0 if not self.use_cuda else h0.cuda()
        return h0


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        out = torch.cat([x, out], 1)
        return out


class TransitionBlock(nn.Module):

    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate,
                            inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):

    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


def densenet_40_12_bc(pretrained=False, requires_grad=False, **kwargs):
    layers = 40
    depth = 10
    growth_rate = 12
    reduce_rate = 0.5
    drop_rate = 0.0
    bottleneck = True
    model = DenseNet3(layers, depth, growth_rate, reduction=reduce_rate,
                      bottleneck=bottleneck, dropRate=drop_rate)
    in_planes = model.in_planes
    if pretrained:
        checkpoint = torch.load(
            densenet_40_12_bc_weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        model.requires_grad = requires_grad

    # Removing linear layer
    removed = list(model.children())[:-1]
    model = torch.nn.Sequential(*removed)
    return model, in_planes


class DenseNet3(nn.Module):

    def __init__(self, depth, num_classes, growth_rate=12, reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(
            math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        # changed this line so avgpool is a layer in the net!!
        self.avpl = nn.AvgPool2d(16)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avpl(out)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


class MRISequenceNet(nn.Module):
    """
    Simple container network for MRI sequence classification. This module consists of:

        1) A frame encoder, e.g., a ConvNet/CNN
        2) A sequence encoder for merging frame representations, e.g., an RNN

    """

    def __init__(self, frame_encoder, seq_encoder, use_cuda=False):
        super(MRISequenceNet, self).__init__()
        self.fenc = frame_encoder
        self.senc = seq_encoder
        self.use_cuda = use_cuda

    def init_hidden(self, batch_size):
        return self.senc.init_hidden(batch_size)

    def embedding(self, x, hidden):
        """Get learned representation of MRI sequence"""
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        batch_size, num_frames, num_channels, width, height = x.size()
        x = x.view(-1, num_channels, width, height)
        x = self.fenc(x)
        x = x.view(batch_size, num_frames, -1)
        x = self.senc.embedding(x, hidden)
        if self.use_cuda:
            return x.cpu()
        else:
            return x

    def forward(self, x, hidden=None):
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()
        # collapse all frames into new batch = batch_size * num_frames
        batch_size, num_frames, num_channels, width, height = x.size()
        x = x.view(-1, num_channels, width, height)
        # encode frames
        x = self.fenc(x)
        x = x.view(batch_size, num_frames, -1)
        # encode sequence
        x = self.senc(x, hidden)
        return x

    def predict_proba(self, data_loader, binary=True, pos_label=1):
        """ Forward inference """
        y_pred = []
        for i, data in enumerate(data_loader):
            x, y = data
            x = Variable(x) if not self.use_cuda else Variable(x).cuda()
            y = Variable(y) if not self.use_cuda else Variable(y).cuda()
            h0 = self.init_hidden(x.size(0))
            outputs = self(x, h0)
            y_hat = F.softmax(outputs, dim=1)
            y_hat = y_hat.data.numpy() if not self.use_cuda else y_hat.cpu().data.numpy()
            y_pred.append(y_hat)
            # empty cuda cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        y_pred = np.concatenate(y_pred)
        return y_pred[:, pos_label] if binary else y_pred

    def predict(self, data_loader, binary=True, pos_label=1, threshold=0.5, return_proba=False, topSelection=None):
        """
        If binary classification, use threshold on positive class
        If multinomial, just select the max probability as the predicted class
        :param data_loader:
        :param binary:
        :param pos_label:
        :param threshold:
        :return:
        """
        proba = self.predict_proba(data_loader, binary, pos_label)
        if topSelection is not None and topSelection < proba.shape[0]:
            threshold = proba[np.argsort(proba)[-topSelection - 1]]
        if binary:
            pred = np.array([1 if p > threshold else 0 for p in proba])
        else:
            pred = np.argmax(proba, 1)

        if return_proba:
            return (proba, pred)
        else:
            return pred


class Dense4012FrameRNN(MRISequenceNet):

    def __init__(self,
                 n_classes,
                 use_cuda=True,
                 input_shape=(3, 130, 130),
                 seq_output_size=128,
                 seq_dropout=0.1,
                 seq_attention=True,
                 seq_bidirectional=True,
                 seq_max_seq_len=25,
                 seq_rnn_type="LSTM",
                 pretrained=True,
                 requires_grad=True):
        super(Dense4012FrameRNN, self).__init__(
            frame_encoder=None, seq_encoder=None, use_cuda=use_cuda)

        self.name = "Dense4012FrameRNN"

        self.fenc, _ = densenet_40_12_bc(
            pretrained=pretrained, requires_grad=requires_grad)
        frm_output_size = self.get_frm_output_size(input_shape)

        self.senc = RNN(n_classes=n_classes,
                        input_size=frm_output_size,
                        hidden_size=seq_output_size,
                        dropout=seq_dropout,
                        max_seq_len=seq_max_seq_len,
                        attention=seq_attention,
                        rnn_type=seq_rnn_type,
                        bidirectional=seq_bidirectional,
                        use_cuda=self.use_cuda)

    def get_frm_output_size(self, input_shape):
        input_shape = list(input_shape)
        input_shape.insert(0, 1)
        dummy_batch_size = tuple(input_shape)
        x = torch.autograd.Variable(torch.zeros(dummy_batch_size))
        frm_output_size = self.fenc.forward(x).view(-1).size()[0]
        return frm_output_size
