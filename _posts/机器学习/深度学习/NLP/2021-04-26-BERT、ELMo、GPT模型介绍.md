---
layout:     post                    # 使用的布局（不需要改）
title:      BERT、ELMo、GPT模型		        # 标题 
subtitle:    	# 副标题
date:       2021-04-26              # 时间
author:     新宇                     # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               # 标签
    - NLP
---
# 一、BERT
## 1. BERT介绍
- BERT是2018年10月由Google AI研究院提出的一种预训练模型.
	- BERT的全称是Bidirectional Encoder Representation from Transformers.
	- BERT在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩: 全部两个衡量指标上全面超越人类, 并且在11种不同NLP测试中创出SOTA表现. 包括将GLUE基准推高至80.4% (绝对改进7.6%), MultiNLI准确度达到86.7% (绝对改进5.6%). 成为NLP发展史上的里程碑式的模型成就.

## 2. BERT的架构
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkbhia4sj30kk0ao0y9.jpg)
- 从上面的架构图中可以看到, 宏观上BERT分三个主要模块.
	- 最底层黄色标记的Embedding模块.
	- 中间层蓝色标记的Transformer模块.
	- 最上层绿色标记的预微调模块

### 2.1 Embedding模块
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkd0e55cj30lm0gbn47.jpg)

### 2.2 双向Transformer模块
BERT中只使用了经典Transformer架构中的Encoder部分, 完全舍弃了Decoder部分. 而两大预训练任务也集中体现在训练Transformer模块中.

### 2.3 预微调模块
- 经过中间层Transformer的处理后, BERT的最后一层根据任务的不同需求而做不同的调整即可.
- 比如对于sequence-level的分类任务, BERT直接取第一个[CLS] token 的final hidden state, 再加一层全连接层后进行softmax来预测最终的标签.

![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkdrfbjhj30ll0oxk6v.jpg)

## 3. BERT的预训练任务
- BERT包含两个预训练任务:
	- **任务一: Masked LM (带mask的语言模型训练)**
		- 关于传统的语言模型训练, 都是采用left-to-right, 或者left-to-right + right-to-left结合的方式, 但这种单向方式或者拼接的方式提取特征的能力有限. 为此BERT提出一个深度双向表达模型(deep bidirectional representation). 即采用MASK任务来训练模型.
		- 1: 在原始训练文本中, 随机的抽取15%的token作为参与MASK任务的对象.
		- 2: 在这些被选中的token中, 数据生成器并不是把它们全部变成[MASK], 而是有下列3种情况.
			- 2.1: 在80%的概率下, 用[MASK]标记替换该token, 比如my dog is hairy -> my dog is [MASK]
			- 2.2: 在10%的概率下, 用一个随机的单词替换token, 比如my dog is hairy -> my dog is apple
			- 2.3: 在10%的概率下, 保持该token不变, 比如my dog is hairy -> my dog is hairy
		- 3: 模型在训练的过程中, 并不知道它将要预测哪些单词? 哪些单词是原始的样子? 哪些单词被遮掩成了[MASK]? 哪些单词被替换成了其他单词? 正是在这样一种高度不确定的情况下, 反倒逼着模型快速学习该token的分布式上下文的语义, 尽最大努力学习原始语言说话的样子. 同时因为原始文本中只有15%的token参与了MASK操作, 并不会破坏原语言的表达能力和语言规则.
	- **任务二: Next Sentence Prediction (下一句话预测任务)**
		- 在NLP中有一类重要的问题比如QA(Quention-Answer), NLI(Natural Language Inference), 需要模型能够很好的理解两个句子之间的关系, 从而需要在模型的训练中引入对应的任务. 在BERT中引入的就是Next Sentence Prediction任务. 采用的方式是输入句子对(A, B), 模型来预测句子B是不是句子A的真实的下一句话.
		- 1: 所有参与任务训练的语句都被选中作为句子A.
			- 1.1: 其中50%的B是原始文本中真实跟随A的下一句话. (标记为IsNext, 代表正样本)
			- 1.2: 其中50%的B是原始文本中随机抽取的一句话. (标记为NotNext, 代表负样本)
		- 2: 在任务二中, BERT模型可以在测试集上取得97%-98%的准确率.

## 4. BERT模型的优点和缺点
- BERT的优点
	- 1: 通过预训练, 加上Fine-tunning, 在11项NLP任务上取得最优结果.
	- 2: BERT的根基源于Transformer, 相比传统RNN更加高效, 可以并行化处理同时能捕捉长距离的语义和结构依赖.
	- 3: BERT采用了Transformer架构中的Encoder模块, 不仅仅获得了真正意义上的bidirectional context, 而且为后续微调任务留出了足够的调整空间.
- BERT的缺点
	- 1: BERT模型过于庞大, 参数太多, 不利于资源紧张的应用场景, 也不利于上线的实时处理.
	- 2: BERT目前给出的中文模型中, 是以字为基本token单位的, 很多需要词向量的应用无法直接使用. 同时该模型无法识别很多生僻词, 只能以UNK代替.
	- 3: BERT中第一个预训练任务MLM中, [MASK]标记只在训练阶段出现, 而在预测阶段不会出现, 这就造成了一定的信息偏差, 因此训练时不能过多的使用[MASK], 否则会影响模型的表现.
	- 4: 按照BERT的MLM任务中的约定, 每个batch数据中只有15%的token参与了训练, 被模型学习和预测, 所以BERT收敛的速度比left-to-right模型要慢很多(left-to-right模型中每一个token都会参与训练).

## 5. MLM任务中采用80%, 10%, 10%策略的原因
- 1: 首先, 如果所有参与训练的token被100%的[MASK], 那么在fine-tunning的时候所有单词都是已知的, 不存在[MASK], 那么模型就只能根据其他token的信息和语序结构来预测当前词, 而无法利用到这个词本身的信息, 因为它们从未出现在训练过程中, 等于模型从未接触到它们的信息, 等于整个语义空间损失了部分信息. 采用80%的概率下应用[MASK], 既可以让模型去学着预测这些单词, 又以20%的概率保留了语义信息展示给模型.
- 2: 保留下来的信息如果全部使用原始token, 那么模型在预训练的时候可能会偷懒, 直接照抄当前token信息. 采用10%概率下random token来随机替换当前token, 会让模型不能去死记硬背当前的token, 而去尽力学习单词周边的语义表达和远距离的信息依赖, 尝试建模完整的语言信息.
- 3: 最后再以10%的概率保留原始的token, 意义就是保留语言本来的面貌, 让信息不至于完全被遮掩, 使得模型可以"看清"真实的语言面貌.

- 用通俗的话理解：
	- BERT中MLM任务中的[MASK]是以一种显示的方式告诉模型"这个词我不告诉你, 你自己从上下文里猜", 非常类似于同学们在做完形填空. 如果[MASK]意外的部分全部都用原始token, 模型会学习到"如果当前词是[MASK], 就根据其他词的信息推断这个词; 如果当前词是一个正常的单词, 就直接照抄". 这样一来, 到了fine-tunning阶段, 所有单词都是正常单词了, 模型就会照抄所有单词, 不再提取单词之间的依赖关系了.
	- BERT中MLM任务以10%的概率填入random token, 就是让模型时刻处于"紧张情绪"中, 让模型搞不清楚当前看到的token是真实的单词还是被随机替换掉的单词, 这样模型在任意的token位置就只能把当前token的信息和上下文信息结合起来做综合的判断和建模. 这样一来, 到了fine-tunning阶段, 模型也会同时提取这两方面的信息, 因为模型"心理很紧张", 它不知道当前看到的这个token, 所谓的"正常单词"到底有没有"提前被动过手脚".

## 6. BERT处理长文本的方法
- 首选要明确一点, BERT预训练模型所接收的最大sequence长度是512.
- 那么对于长文本(文本长度超过512的句子), 就需要特殊的方式来构造训练样本. 核心就是如何进行截断.
	- 1: head-only方式: 这是只保留长文本头部信息的截断方式, 具体为保存前510个token (要留两个位置给[CLS]和[SEP]).
	- 2: tail-only方式: 这是只保留长文本尾部信息的截断方式, 具体为保存最后510个token (要留两个位置给[CLS]和[SEP]).
	- 3: head+only方式: 选择前128个token和最后382个token (文本总长度在800以内), 或者前256个token和最后254个token (文本总长度大于800).

# 二、ELMo
## 1. 什么是ELMo
- ELMo是2018年3月由华盛顿大学提出的一种预训练模型.
	- ELMo的全称是Embeddings from Language Models.
	- ELMo模型的提出源于论文[Deep Contextualized Word Representations](https://arxiv.org/abs/1802.05365)
	- ELMo模型提出的动机源于研究人员认为一个好的预训练语言模型应该能够包含丰富的句法和语义信息, 并且能够对多义词进行建模. 而传统的词向量(2013年的word2vec, 2014年的GloVe)都是上下文无关的, 也就是固定的词向量. 最典型的例子就是"apple"在不同的语境下, 应该可以表示水果或公司, 但是固定的词向量显然无法做到这一点. 因为研究团队利用新的语言模型训练一个上下文相关的预训练模型, 成为ELMo, 并在6个NLP任务上获得提升.


## 2. ELMo的架构
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxknpzbuqj30lc0h2tex.jpg)
- Embedding模块: ELMo最底层的词嵌入采用CNN对字符级进行编码, 本质就是获得一个静态的词嵌入向量作为网络的底层输入.
- 两部分的双层LSTM模块:
	- 这是整个ELMo中最重要的部分, 架构中分成左侧的前向LSTM网络, 和右侧的反向LSTM网络.
	- ELMo的做法是我们只预训练一个Language Model, 而word embedding是通过输入的句子实时给出的, 这样单词的嵌入向量就包含了上下文的信息, 也就彻底改变了Word2Vec和GloVe的静态词向量的做法.

![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkot43lbj30lo0kvtbp.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkp6huqdj30lj0ju41y.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkpg0etpj30lk08wwgk.jpg)

## 3. ELMo的预训练任务
- ELMo的本质思想:
	- 首先用一个语言模型学好一个单词的word embedding, 此时是无法区分多义词的, 但没关系. 当实际使用word embedding的时候, 该单词已经具备了特定的上下文信息, 这个时候可以根据上下文单词的语义去调整单词的word embedding表示, 这样经过调整后得到的word embedding向量就可以准确的表达单词在当前上下文中的真实含义了, 也就自然的解决了多义词问题.
	- 结论就是ELMo模型是个根据当前上下文对word embedding动态调整的语言模型.
- ELMo的预训练采用了典型的两阶段过程:
	- 第一阶段: 利用语言模型进行预训练.
	- 第二阶段: 在做下游任务时, 从预训练网络中提取对应单词的网络各层的word embedding作为新特征补充到下游任务中.

![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkqfsr1dj30m60jrtcl.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkqn4vxnj30m10atdhx.jpg)

## 4. ELMo模型的效果
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkqx5nrqj30ly0n247t.jpg)

## 5. ELMo的待改进点
- ELMo在传统静态word embedding方法(Word2Vec, GloVe)的基础上提升了很多, 但是依然存在缺陷, 有很大的改进余地.
	- 第一点: 一个很明显的缺点在于特征提取器的选择上, ELMo使用了双向双层LSTM, 而不是现在横扫千军的Transformer, 在特征提取能力上肯定是要弱一些的. 设想如果ELMo的提升提取器选用Transformer, 那么后来的BERT的反响将远不如当时那么火爆了.
	- 第二点: ELMo选用双向拼接的方式进行特征融合, 这种方法肯定不如BERT一体化的双向提取特征好.

# 三、GPT 
## 1. 什么是GPT
- GPT是OpenAI公司提出的一种语言预训练模型.
	- OpenAI在论文[Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)中提出GPT模型.
	- OpenAI后续又在论文[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)中提出GPT2模型.
	- GPT和GPT2模型结构差别不大, 但是GPT2采用了更大的数据集进行训练.
- OpenAI GPT模型是在Google BERT模型之前提出的, 与BERT最大的区别在于GPT采用了传统的语言模型方法进行预训练, 即使用单词的上文来预测单词, 而BERT是采用了双向上下文的信息共同来预测单词.
	- 正是因为训练方法上的区别, 使得GPT更擅长处理自然语言生成任务(NLG), 而BERT更擅长处理自然语言理解任务(NLU).

## 2. GPT的架构
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkubnd58j30le0evgrl.jpg)

作为两大模型的直接对比, BERT采用了Transformer的Encoder模块, 而GPT采用了Transformer的Decoder模块. 并且GPT的Decoder Block和经典Transformer Decoder Block还有所不同, 如下图所示:
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkwfb64qj30gr0a8q66.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkwrtcfsj30lk0mln65.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkx1pd43j30lp079abi.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkxvy2hlj30ed090gmw.jpg)

## 3. GPT的训练过程
- GPT的训练也是典型的两阶段过程:
	- 第一阶段: 无监督的预训练语言模型.
	- 第二阶段: 有监督的下游任务fine-tunning.

![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkyflghrj30mv0imacn.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkyo9rw5j30lu0d4mz3.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkyvshjlj30m70g5wh6.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkz28dooj30lc07wjs1.jpg)

# 四、GPT2
## 1. GPT2的架构
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkzoxq9ij30lz0jfgrv.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxkzvye4qj30lm0h1n3e.jpg)

## 2. GPT2的模型细节
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl079nmnj30lq0fldj0.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl0g0dn3j30la0hb77f.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl0nrxdgj30lh0c8q56.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl0w8llpj30ld0ixjwu.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl17csamj30lq0lfq7e.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl1hyh7wj30lj0hb777.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl1sk9nqj30lo0o60xb.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl20omryj30ld0cmmyn.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl27j4k2j30ll0d940z.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl2gb5wyj30lg0i1n0h.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl2rm75fj30l50i30w0.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl322ewmj30l50d6tar.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl3dhb19j30lm0j643x.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl3lezr8j30ld0gntdm.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl3t455pj30l80icwj6.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl41di3uj30lf0f2myo.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl4a1acqj30la0clq5a.jpg)
![](https://tva1.sinaimg.cn/large/008i3skNly1gpxl4ilbx3j30ll0jvn26.jpg)

# 五、BERT, GPT, ELMo模型对比
- BERT, GPT, ELMo之间的不同点
	- 关于特征提取器:
		- ELMo采用两部分双层双向LSTM进行特征提取, 然后再进行特征拼接来融合语义信息.
		- GPT和BERT采用Transformer进行特征提取.
			- BERT采用的是Transformer架构中的Encoder模块.
			- GPT采用的是Transformer架构中的Decoder模块.
		- 很多NLP任务表明Transformer的特征提取能力强于LSTM, 对于ELMo而言, 采用1层静态token embedding + 2层LSTM, 提取特征的能力有限.
	- 单/双向语言模型:
		- 三者之中, 只有GPT采用单向语言模型, 而ELMo和BERT都采用双向语言模型.
		- ELMo虽然被认为采用了双向语言模型, 但实际上是左右两个单向语言模型分别提取特征, 然后进行特征拼接, 这种融合特征的能力比BERT一体化的融合特征方式弱.
		- 三者之中, 只有ELMo没有采用Transformer. GPT和BERT都源于Transformer架构, GPT的单向语言模型采用了经过修改后的Decoder模块, Decoder采用了look-ahead mask, 只能看到context before上文信息, 未来的信息都被mask掉了. 而BERT的双向语言模型采用了Encoder模块, Encoder只采用了padding mask, 可以同时看到context before上文信息, 以及context after下文信息.
- BERT, GPT, ELMo各自的优点和缺点
	- ELMo:
		- 优点:
			- 从早期的Word2Vec预训练模型的最大缺点出发, 进行改进, 这一缺点就是无法解决多义词的问题.
			- ELMo根据上下文动态调整word embedding, 可以解决多义词的问题.
		- 缺点:
			- ELMo使用LSTM提取特征的能力弱于Transformer.
			- ELMo使用向量拼接的方式融合上下文特征的能力弱于Transformer.
	- GPT:
		- 优点:
			- GPT使用了Transformer提取特征, 使得模型能力大幅提升.
		- 缺点:
			- GPT只使用了单向Decoder, 无法融合未来的信息.
	- BERT:
		- 优点:
			- BERT使用了双向Transformer提取特征, 使得模型能力大幅提升.
			- 添加了两个预训练任务, MLM + NSP的多任务方式进行模型预训练.
		- 缺点:
			- 模型过于庞大, 参数量太多, 需要的数据和算力要求过高, 训练好的模型应用场景要求高.
			- 更适合用于语言嵌入表达, 语言理解方面的任务, 不适合用于生成式的任务.