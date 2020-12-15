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

import time


batch_size = 1

# 模型参数
model_name = 'TextCNN' # 模型名
class_num = 49
kernel_size = [5] * 4
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

        dim = int(sum(channel_sizes[1:]))
        self.classifer = torch.nn.Sequential(torch.nn.Linear(in_features=dim, out_features=dim),
            torch.nn.Tanh(),
            torch.nn.Linear(in_features=dim, out_features=2))
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(0)
        x = torch.transpose(x, 1, 2)

        outputs = []
        for conv in self.convs:
            x = F.relu(conv(x))
            outputs.append(x.max(2)[0])

        x = torch.cat(outputs, dim=1)
        logits = [self.classifer(x) for i in range(class_num)]
        ret = torch.cat(logits, dim=-1)
        return ret

if __name__ == '__main__':
    model = ModelCNN()
    model.to(torch.device("cpu"))
    data  = torch.LongTensor([i for i in range(150)])
    ret = model.forward(data)
    print(ret)
    start = time.time()
    for i in range(1000):
        model(data)
    print("Latency(ms):", time.time()-start)
    with torch.autograd.profiler.profile(use_cuda=False, record_shapes=True) as prof:
        model.forward(data)

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))
