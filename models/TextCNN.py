'''
@Author: Gordon Lee
@Date: 2019-08-09 16:29:55
@LastEditors: Gordon Lee
@LastEditTime: 2019-08-16 19:00:19
@Description: 
'''
import json
import torch
import torch.nn as nn
import torch.nn.functional as F




batch_size = 1

# 模型参数
model_name = 'TextCNN' # 模型名
class_num = 49
kernel_size = 5
vocab_size = 100000
embed_dim = 256 # 未使用预训练词向量的默认值

class ModelCNN(nn.Module):
    '''
    TextCNN: CNN-rand, CNN-static, CNN-non-static, CNN-multichannel
    '''
    def __init__(self):

        super(ModelCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        channel_sizes = [256, 384, 256, 256, 256]
        self.convs = nn.ModuleList([
            nn.Conv1d(channel_num, kernel_num, kernel_size, padding=kernel_size//2) 
            for channel_num, kernel_num, kernel_size in zip(channel_sizes[:-1], channel_sizes[1:], kernel_size)
        ])

        dim = int(sum(channel_size[1:]))
        # 全连接层
        self.classifer = torch.nn.Sequential(torch.nn.Linear(in_features=dim, out_features=dim),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=dim, out_features=2))
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        # 池化
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x] 
        # flatten
        x = torch.cat(x, 1) 
        logits = [self.classifer(x) for i in range(class_num)]
        return torch.cat(logits, dim=-1)

if __name__ == '__main__':
    model = ModelCNN()
    ret = model.forward()