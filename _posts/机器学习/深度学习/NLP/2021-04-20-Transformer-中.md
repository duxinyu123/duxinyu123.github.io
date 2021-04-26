---
layout:     post                    # 使用的布局（不需要改）
title:      Transformer-中		        # 标题 
subtitle:   解码器、输出部分、模型构建 	# 副标题
date:       2021-04-18              # 时间
author:     新宇                     # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               # 标签
    - NLP
---
# 一、解码器
- 解码器部分:
	- 由N个解码器层堆叠而成
	- 每个解码器层由三个子层连接结构组成
	- 第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
	- 第二个子层连接结构包括一个多头注意力子层和规范化层以及一个残差连接
	- 第三个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连接

![](https://tva1.sinaimg.cn/large/008i3skNly1gpxcr3v9rqj30m20degn4.jpg)
## 1. 解码器层
```python
# 使用DecoderLayer的类实现解码器层

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """初始化函数的参数有5个, 分别是size，代表词嵌入的维度大小, 同时也代表解码器层的尺寸，
            第二个是self_attn，多头自注意力对象，也就是说这个注意力机制需要Q=K=V， 
            第三个是src_attn，多头注意力对象，这里Q!=K=V， 第四个是前馈全连接层对象，最后就是droupout置0比率.
        """

        super(DecoderLayer, self).__init__()
        # 在初始化函数中， 主要就是将这些输入传到类中

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 按照结构图使用clones函数克隆三个子层连接对象.

        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        """forward函数中的参数有4个，分别是来自上一层的输入x，
           来自编码器层的语义存储变量mermory， 以及源数据掩码张量和目标数据掩码张量.
        """

        # 将memory表示成m方便之后使用

        m = memory

        # 将x传入第一个子层结构，第一个子层结构的输入分别是x和self-attn函数，因为是自注意力机制，所以Q,K,V都是x，

        # 最后一个参数是目标数据掩码张量，这时要对目标数据进行遮掩，因为此时模型可能还没有生成任何目标数据，

        # 比如在解码器准备生成第一个字符或词汇时，我们其实已经传入了第一个字符以便计算损失，

        # 但是我们不希望在生成第一个字符时模型能利用这个信息，因此我们会将其遮掩，同样生成第二个字符或词汇时，

        # 模型只能使用第一个字符或词汇信息，第二个字符以及之后的信息都不允许被模型使用.

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))

        # 接着进入第二个子层，这个子层中常规的注意力机制，q是输入x; k，v是编码层输出memory， 

        # 同样也传入source_mask，但是进行源数据遮掩的原因并非是抑制信息泄漏，而是遮蔽掉对结果没有意义的字符而产生的注意力值，

        # 以此提升模型效果和训练速度. 这样就完成了第二个子层的处理.

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))

        # 最后一个子层就是前馈全连接子层，经过它的处理后就可以返回结果.这就是我们的解码器层结构.

        return self.sublayer[2](x, self.feed_forward)
```

## 2. 解码器
```python
# 使用类Decoder来实现解码器

class Decoder(nn.Module):
    def __init__(self, layer, N):
        """初始化函数的参数有两个，第一个就是解码器层layer，第二个是解码器层的个数N."""

        super(Decoder, self).__init__()
        # 首先使用clones方法克隆了N个layer，然后实例化了一个规范化层. 

        # 因为数据走过了所有的解码器层后最后要做规范化处理. 

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        """forward函数中的参数有4个，x代表目标数据的嵌入表示，memory是编码器层的输出，
           source_mask, target_mask代表源数据和目标数据的掩码张量"""

        # 然后就是对每个层进行循环，当然这个循环就是变量x通过每一个层的处理，

        # 得出最后的结果，再进行一次规范化返回即可. 

        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)
```

# 二、输出部分
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxctawprqj30lw0fkq44.jpg)
```python
# nn.functional工具包装载了网络层中那些只进行计算, 而没有参数的层

import torch.nn.functional as F

# 将线性层和softmax计算层一起实现, 因为二者的共同目标是生成最后的结构

# 因此把类的名字叫做Generator, 生成器类

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        """初始化函数的输入参数有两个, d_model代表词嵌入维度, vocab_size代表词表大小."""

        super(Generator, self).__init__()
        # 首先就是使用nn中的预定义线性层进行实例化, 得到一个对象self.project等待使用, 

        # 这个线性层的参数有两个, 就是初始化函数传进来的两个参数: d_model, vocab_size

        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """前向逻辑函数中输入是上一层的输出张量x"""

        # 在函数中, 首先使用上一步得到的self.project对x进行线性变化, 

        # 然后使用F中已经实现的log_softmax进行的softmax处理.

        # 在这里之所以使用log_softmax是因为和我们这个pytorch版本的损失函数实现有关, 在其他版本中将修复.

        # log_softmax就是对softmax的结果又取了对数, 因为对数函数是单调递增函数, 

        # 因此对最终我们取最大的概率值没有影响. 最后返回结果即可.

        return F.log_softmax(self.project(x), dim=-1)
```

# 三、模型构建
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxcuh72aaj30ku0gyjtq.jpg)

## 1. 编码器解码器实现
```python
# 使用EncoderDecoder类来实现编码器-解码器结构

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        """初始化函数中有5个参数, 分别是编码器对象, 解码器对象, 
           源数据嵌入函数, 目标数据嵌入函数,  以及输出部分的类别生成器对象
        """

        super(EncoderDecoder, self).__init__()
        # 将参数传入到类中

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        """在forward函数中，有四个参数, source代表源数据, target代表目标数据, 
           source_mask和target_mask代表对应的掩码张量"""

        # 在函数中, 将source, source_mask传入编码函数, 得到结果后,

        # 与source_mask，target，和target_mask一同传给解码函数.

        return self.decode(self.encode(source, source_mask), source_mask,
                            target, target_mask)

    def encode(self, source, source_mask):
        """编码函数, 以source和source_mask为参数"""

        # 使用src_embed对source做处理, 然后和source_mask一起传给self.encoder

        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        """解码函数, 以memory即编码器的输出, source_mask, target, target_mask为参数"""

        # 使用tgt_embed对target做处理, 然后和source_mask, target_mask, memory一起传给self.decoder

        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)
```

## 2. 模型构建
```python
def make_model(source_vocab, target_vocab, N=6, 
               d_model=512, d_ff=2048, head=8, dropout=0.1):
    """该函数用来构建模型, 有7个参数，分别是源数据特征(词汇)总数，目标数据特征(词汇)总数，
       编码器和解码器堆叠数，词向量映射维度，前馈全连接网络中变换矩阵的维度，
       多头注意力结构中的多头数，以及置零比率dropout."""

    # 首先得到一个深度拷贝命令，接下来很多结构都需要进行深度拷贝，

    # 来保证他们彼此之间相互独立，不受干扰.

    c = copy.deepcopy

    # 实例化了多头注意力类，得到对象attn

    attn = MultiHeadedAttention(head, d_model)

    # 然后实例化前馈全连接类，得到对象ff 

    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # 实例化位置编码类，得到对象position

    position = PositionalEncoding(d_model, dropout)

    # 根据结构图, 最外层是EncoderDecoder，在EncoderDecoder中，

    # 分别是编码器层，解码器层，源数据Embedding层和位置编码组成的有序结构，

    # 目标数据Embedding层和位置编码组成的有序结构，以及类别生成器层. 

    # 在编码器层中有attention子层以及前馈全连接子层，

    # 在解码器层中有两个attention子层以及前馈全连接层.

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, source_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, target_vocab), c(position)),
        Generator(d_model, target_vocab))

    # 模型结构完成后，接下来就是初始化模型中的参数，比如线性层中的变换矩阵

    # 这里一但判断参数的维度大于1，则会将其初始化成一个服从均匀分布的矩阵，

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```

# 四、模型基本测试运行
- 使用copy任务进行模型基本测试的四步曲
	- 第一步: 构建数据集生成器
	- 第二步: 获得Transformer模型及其优化器和损失函数
	- 第三步: 运行模型进行训练和评估
	- 第四步: 使用模型进行贪婪解码

## 1. 构建数据集生成器

```python
# 导入工具包Batch, 它能够对原始样本数据生成对应批次的掩码张量
from pyitcast.transformer_utils import Batch  

def data_generator(V, batch, num_batch):
    """该函数用于随机生成copy任务的数据, 它的三个输入参数是V: 随机生成数字的最大值+1, 
       batch: 每次输送给模型更新一次参数的数据量, num_batch: 一共输送num_batch次完成一轮
    """
    # 使用for循环遍历nbatches
    for i in range(num_batch):
        # 在循环中使用np的random.randint方法随机生成[1, V)的整数, 
        # 分布在(batch, 10)形状的矩阵中, 然后再把numpy形式转换称torch中的tensor.
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))

        # 接着使数据矩阵中的第一列数字都为1, 这一列也就成为了起始标志列, 
        # 当解码器进行第一次解码的时候, 会使用起始标志列作为输入.
        data[:, 0] = 1

        # 因为是copy任务, 所有source与target是完全相同的, 且数据样本作用变量不需要求梯度
        # 因此requires_grad设置为False
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)

        # 使用Batch对source和target进行对应批次的掩码张量生成, 最后使用yield返回
        yield Batch(source, target) 
```

## 2. 获得Transformer模型及其优化器和损失函数

```python
# 导入优化器工具包get_std_opt, 该工具用于获得标准的针对Transformer模型的优化器

# 该标准优化器基于Adam优化器, 使其对序列到序列的任务更有效.

from pyitcast.transformer_utils import get_std_opt

# 导入标签平滑工具包, 该工具用于标签平滑, 标签平滑的作用就是小幅度的改变原有标签值的值域

# 因为在理论上即使是人工的标注数据也可能并非完全正确, 会受到一些外界因素的影响而产生一些微小的偏差

# 因此使用标签平滑来弥补这种偏差, 减少模型对某一条规律的绝对认知, 以防止过拟合. 通过下面示例了解更多.

from pyitcast.transformer_utils import LabelSmoothing

# 导入损失计算工具包, 该工具能够使用标签平滑后的结果进行损失的计算, 

# 损失的计算方法可以认为是交叉熵损失函数.

from pyitcast.transformer_utils import SimpleLossCompute

# 使用make_model获得model

model = make_model(V, V, N=2)

# 使用get_std_opt获得模型优化器

model_optimizer = get_std_opt(model)

# 使用LabelSmoothing获得标签平滑对象

criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

# 使用SimpleLossCompute获得利用标签平滑结果的损失计算方法

loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


# 模型平滑示例

from pyitcast.transformer_utils import LabelSmoothing

# 使用LabelSmoothing实例化一个crit对象.

# 第一个参数size代表目标数据的词汇总数, 也是模型最后一层得到张量的最后一维大小

# 这里是5说明目标词汇总数是5个. 第二个参数padding_idx表示要将那些tensor中的数字

# 替换成0, 一般padding_idx=0表示不进行替换. 第三个参数smoothing, 表示标签的平滑程度

# 如原来标签的表示值为1, 则平滑后它的值域变为[1-smoothing, 1+smoothing].

crit = LabelSmoothing(size=5, padding_idx=0, smoothing=0.5)

# 假定一个任意的模型最后输出预测结果和真实结果

predict = Variable(torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
                             [0, 0.2, 0.7, 0.1, 0], 
                             [0, 0.2, 0.7, 0.1, 0]]))

# 标签的表示值是0，1，2

target = Variable(torch.LongTensor([2, 1, 0]))

# 将predict, target传入到对象中

crit(predict, target)

# 绘制标签平滑图像

plt.imshow(crit.true_dist)
```
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxcxyo05xj30jr0deaa8.jpg)

## 3. 运行模型进行训练和评估
```python
# 导入模型单轮训练工具包run_epoch, 该工具将对模型使用给定的损失函数计算方法进行单轮参数更新.
# 并打印每轮参数更新的损失结果.
from pyitcast.transformer_utils import run_epoch

def run(model, loss, epochs=10):
    """模型训练函数, 共有三个参数, model代表将要进行训练的模型
       loss代表使用的损失计算方法, epochs代表模型训练的轮数"""

    # 遍历轮数
    for epoch in range(epochs):
        # 模型使用训练模式, 所有参数将被更新
        model.train()
        # 训练时, batch_size是20
        run_epoch(data_generator(V, 8, 20), model, loss)

        # 模型使用评估模式, 参数将不会变化 
        model.eval()
        # 评估时, batch_size是5
        run_epoch(data_generator(V, 8, 5), model, loss)
```

## 4. 使用模型进行贪婪解码

```python
# 导入贪婪解码工具包greedy_decode, 该工具将对最终结进行贪婪解码

# 贪婪解码的方式是每次预测都选择概率最大的结果作为输出, 

# 它不一定能获得全局最优性, 但却拥有最高的执行效率.

from pyitcast.transformer_utils import greedy_decode 


def run(model, loss, epochs=10):
    for epoch in range(epochs):
        model.train()

        run_epoch(data_generator(V, 8, 20), model, loss)

        model.eval()

        run_epoch(data_generator(V, 8, 5), model, loss)

    # 模型进入测试模式

    model.eval()

    # 假定的输入张量

    source = Variable(torch.LongTensor([[1,3,2,5,4,6,7,8,9,10]]))

    # 定义源数据掩码张量, 因为元素都是1, 在我们这里1代表不遮掩

    # 因此相当于对源数据没有任何遮掩.

    source_mask = Variable(torch.ones(1, 1, 10))

    # 最后将model, src, src_mask, 解码的最大长度限制max_len, 默认为10

    # 以及起始标志数字, 默认为1, 我们这里使用的也是1
    
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == '__main__':
    run(model, loss) 
```




