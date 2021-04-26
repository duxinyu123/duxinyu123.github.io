---
layout:     post                    # 使用的布局（不需要改）
title:      Transformer-下		        # 标题 
subtitle:   Transformer经典案例 	# 副标题
date:       2021-04-21              # 时间
author:     新宇                     # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               # 标签
    - NLP
---
# 一、案例介绍
- 什么是语言模型:
	- 以一个符合语言规律的序列为输入，模型将利用序列间关系等特征，输出一个在所有词汇上的概率分布.这样的模型称为语言模型.
	- ![](https://tva1.sinaimg.cn/large/008i3skNly1gpxd0vu9zzj30hw02h74l.jpg)

- 语言模型能解决哪些问题:
	- 根据语言模型的定义，可以在它的基础上完成机器翻译，文本生成等任务，因为我们通过最后输出的概率分布来预测下一个词汇是什么.
	- 语言模型可以判断输入的序列是否为一句完整的话，因为我们可以根据输出的概率分布查看最大概率是否落在句子结束符上，来判断完整性.
	- 语言模型本身的训练目标是预测下一个词，因为它的特征提取部分会抽象很多语言序列之间的关系，这些关系可能同样对其他语言类任务有效果.因此可以作为预训练模型进行迁移学习.
- 整个案例的实现可分为以下五个步骤
	- 第一步: 导入必备的工具包
	- 第二步: 导入wikiText-2数据集并作基本处理
	- 第三步: 构建用于模型输入的批次化数据
	- 第四步: 构建训练和评估函数
	- 第五步: 进行训练和评估(包括验证以及测试)

# 二、代码实现
## 1. 导入必备工具包
```python
# 数学计算工具包math

import math

# torch以及torch.nn, torch.nn.functional

import torch
import torch.nn as nn
import torch.nn.functional as F

# torch中经典文本数据集有关的工具包

# 具体详情参考下方torchtext介绍

import torchtext

# torchtext中的数据处理工具, get_tokenizer用于英文分词

from torchtext.data.utils import get_tokenizer

# 已经构建完成的TransformerModel

from pyitcast.transformer import TransformerModel
```

## 2. 导入wikiText-2数据集并作基本处理
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxd2xtnypj30m80chtbe.jpg)

```python
# 创建语料域, 语料域是存放语料的数据结构, 

# 它的四个参数代表给存放语料（或称作文本）施加的作用. 

# 分别为 tokenize,使用get_tokenizer("basic_english")获得一个分割器对象,

# 分割方式按照文本为基础英文进行分割. 

# init_token为给文本施加的起始符 <sos>给文本施加的终止符<eos>, 

# 最后一个lower为True, 存放的文本字母全部小写.

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

# 最终获得一个Field对象.

# <torchtext.data.field.Field object at 0x7fc42a02e7f0>

# 然后使用torchtext的数据集方法导入WikiText2数据, 

# 并切分为对应训练文本, 验证文本，测试文本, 并对这些文本施加刚刚创建的语料域.

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

# 我们可以通过examples[0].text取出文本对象进行查看.

# >>> test_txt.examples[0].text[:10]

# ['<eos>', '=', 'robert', '<unk>', '=', '<eos>', '<eos>', 'robert', '<unk>', 'is']

# 将训练集文本数据构建一个vocab对象, 

# 这样可以使用vocab对象的stoi方法统计文本共包含的不重复词汇总数.

TEXT.build_vocab(train_txt)

# 然后选择设备cuda或者cpu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## 3. 构建用于模型输入的批次化数据
```python
def batchify(data, bsz):
    """batchify函数用于将文本数据映射成连续数字, 并转换成指定的样式, 指定的样式可参考下图.
       它有两个输入参数, data就是我们之前得到的文本数据(train_txt, val_txt, test_txt),
       bsz是就是batch_size, 每次模型更新参数的数据量"""
    # 使用TEXT的numericalize方法将单词映射成对应的连续数字.

    data = TEXT.numericalize([data.examples[0].text])
    '''
    >>> data
    tensor([[   3],
       [  12],
       [3852],
       ...,
       [   6],
       [   3],
        [   3]])
    '''

    # 接着用数据词汇总数除以bsz,

    # 取整数得到一个nbatch代表需要多少次batch后能够遍历完所有数据

    nbatch = data.size(0) // bsz

    # 之后使用narrow方法对不规整的剩余数据进行删除,

    # 第一个参数是代表横轴删除还是纵轴删除, 0为横轴，1为纵轴

    # 第二个和第三个参数代表保留开始轴到结束轴的数值.类似于切片

    # 可参考下方演示示例进行更深理解.

    data = data.narrow(0, 0, nbatch * bsz)
    # >>> data
    # tensor([[   3],
    #    [  12],
    #    [3852],
    #    ...,
    #    [  78],
    #    [ 299],
    #    [  36]])
    # 后面不能形成bsz个的一组数据被删除

    # 接着我们使用view方法对data进行矩阵变换, 使其成为如下样式:

    '''
    tensor([[    3,    25,  1849,  ...,     5,    65,    30],
       [   12,    66,    13,  ...,    35,  2438,  4064],
       [ 3852, 13667,  2962,  ...,   902,    33,    20],
       ...,
       [  154,     7,    10,  ...,     5,  1076,    78],
       [   25,     4,  4135,  ...,     4,    56,   299],
       [    6,    57,   385,  ...,  3168,   737,    36]])
    '''

    # 因为会做转置操作, 因此这个矩阵的形状是[None, bsz],

    # 如果输入是训练数据的话，形状为[104335, 20], 可以通过打印data.shape获得.

    # 也就是data的列数是等于bsz的值的.

    data = data.view(bsz, -1).t().contiguous()
    # 最后将数据分配在指定的设备上.

    return data.to(device)
```
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxd50f8u0j30j2089t99.jpg)

torch.narrow演示
```python
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> x.narrow(0, 0, 2)
tensor([[ 1,  2,  3],
        [ 4,  5,  6]])
>>> x.narrow(1, 1, 2)
tensor([[ 2,  3],
        [ 5,  6],
        [ 8,  9]])
```

使用batchify来处理训练数据，验证数据以及测试数据
```python
# 训练数据的batch size

batch_size = 20

# 验证和测试数据（统称为评估数据）的batch size

eval_batch_size = 10

# 获得train_data, val_data, test_data

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)
```
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxd6jqqlmj30mg0i6gog.jpg)

批次化过程的第二个函数get_batch代码分析:
```python
# 令子长度允许的最大值bptt为35

bptt = 35

def get_batch(source, i):
    """用于获得每个批次合理大小的源数据和目标数据.
       参数source是通过batchify得到的train_data/val_data/test_data.
       i是具体的批次次数.
    """

    # 首先我们确定句子长度, 它将是在bptt和len(source) - 1 - i中最小值

    # 实质上, 前面的批次中都会是bptt的值, 只不过最后一个批次中, 句子长度

    # 可能不够bptt的35个, 因此会变为len(source) - 1 - i的值.

    seq_len = min(bptt, len(source) - 1 - i)

    # 语言模型训练的源数据的第i批数据将是batchify的结果的切片[i:i+seq_len]

    data = source[i:i+seq_len]

    # 根据语言模型训练的语料规定, 它的目标数据是源数据向后移动一位

    # 因为最后目标数据的切片会越界, 因此使用view(-1)来保证形状正常.

    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
```


## 4. 构建训练和评估函数

## 5. 进行训练和评估(包括验证以及测试)