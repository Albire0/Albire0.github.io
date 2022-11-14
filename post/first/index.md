# First


# NLP课程详细设计-报告

## 环境准备

Windows10

Pycharm2021

python3.9

剩余的依赖由文件夹下的requirements.txt提供,为以下这些

```
jieba==0.42.1
keras==2.8.0
Keras_Preprocessing==1.1.2
matplotlib==3.5.2
numpy==1.21.4
pandas==1.3.5
scikit_learn==1.1.1
tqdm==4.62.3
gensim==4.1.2
```

数据集选用美团的ASAP-SENT数据集

## 设计流程

对于一个字，词的情感分析模型，设计实验比较字的效果是否好于词，流程为

1. 数据处理
   1. 读取文件中的文本和对应的标签，最大文本的长度
   2. 观察标签的分布情况，发现3，2，1的数量要明显小于4，5
   3. 将3，2，1集合为1类，4，5分别为第2，第3类
   4. 按字/词的方式，构造word2id字典
   5. 将文本中的字/词按word2id字典做映射，截断到最大文本长
2. 数据处理（词向量）
   1. 分词后训练Word2Vec模型
   2. 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
   3. 准备词向量矩阵
3. 搭建模型
   1. 按照下面的模型结构搭建
   2. 模型的优化器选用adam，学习率默认
   3. 模型的评估方法为loss和acc
   4. 模型的损失函数为稀疏交叉熵
4. 训练模型
   1. 训练集和验证集的比例为9：1
   2. 直接将x，y整体输入模型中，不使用生成器
5. 对每个模型绘制loss和acc的图
6. 观察结果，分析

## 数据处理

数据的处理有两种情况，字的部分通过data.py文件中的函数，词的处理通过MyTokenizer类

在data.py中，描述如下

| 作用                              | 函数名                                                     |
| --------------------------------- | ---------------------------------------------------------- |
| 将句子转化为token                 | def sentence_to_token(sentence, word2token: dict):->tokens |
| 对每一条句子做补0或截断操作       | def padding_or_truncate_text(item):->texts                 |
| 读取训练数据                      | def read_train_data()                                      |
| 获得各项分类的权重                | def get_classweights(y:list[int]):                         |
| 获取word2id的字典                 | def get_word2id_json(sen_list):                            |
| 将所有的星级转化为标签分类        | def get_labels(stars):                                     |
| 将好评的星级转化为3个分类中的一个 | def to_label(star):                                        |
| 统计各个star的出现频率            | def calculate_star(stars):                                 |
| 将所有句子转化为tokens            | def get_result(text_a, word2token):                        |

MyTokenizer类中，描述如下

初始化函数

```python
 def __init__(self, char_level, max_word=-1, oov_token="<unk>", lower=True,
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
        """
        Tokenizer的初始化,返回单例
        :param char_level:False按词去分,True 按字去分
        :param max_word:最多分词的个数,默认是为-1,是所有
        :param oov_token:out of vocab的单词对应的token
        :param lower: 是否全部转化为小写 默认为True
        :param filters: filters字符串内,是所有不参与分词的字符 默认为 '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
```

单例构造

```python
    def __new__(cls, *args, **kwargs):
        """
        借助new方法 实现单例模式,这是因为new执行后,才是init执行
        @:return 返回MyTokenizer单例
```

获取Mytokenizer的构造参数

```python
 def get_config(self):
        """
        获取Tokenizer的组装属性,并打印
        :return:返回Tokenizer的组装属性,type:dict
```

把所有的句子变成word_level的

```python
    def text_to_wordlevel(self,texts:list[str]):
        """
        把普通的中文文本,转换为以词为单位,并保存
        :param texts: :param texts: 文章 shape为(list[sentence])
        比如:["飘渺孤鸿影，漏断人初静",”想当年，金戈铁马，气吞万里如虎“]
        :return:以词为单位的文本新数组
        """
        if os.path.exists("word_level_input.txt"):
            f = open("word_level_input.txt", encoding='utf-8', mode='r')
            buffers = f.readlines()
            new_texts = [item.strip("\n") for item in buffers]
        else:
            f = open("word_level_input.txt", encoding='utf-8', mode='w')
            new_texts = []
            for index in tqdm(range(0, len(texts))):
                transformed_text = " ".join(jieba.lcut(texts[index]))
                f.writelines(transformed_text+"\n")
                new_texts.append(transformed_text)
        f.close()
        return new_texts
```

利用text初始化tokenizer，得到字典

```python
    def fit_on_texts(self, texts: list[str]):
        """
        tokenizer获取对应的word2id,和id2word
        :param texts: 文章 shape为(list[sentence])
        比如:["飘渺孤鸿影，漏断人初静",”想当年，金戈铁马，气吞万里如虎“]
        :return:word2id:dict,id2word:dict
        """
        if not isinstance(texts, list) or not isinstance(texts[0], str):
            raise Exception("texts must be a list[str] type")

        if not self.__char_level:
            new_texts = self.text_to_wordlevel(texts)
            self.__TOKENIZER.fit_on_texts(new_texts)
        else:
            self.__TOKENIZER.fit_on_texts(texts)

        self.word2id = self.__TOKENIZER.word_index
        self.id2word = self.__TOKENIZER.index_word

        with open("model/word2id.json", mode='w', encoding='utf-8') as f:
            json.dump(self.word2id, f)

        print("word_dict length:", len(self.word2id))
        print("word_dict saved in word2id.json")
        return self.word2id, self.id2word
```

把所有的句子从词的层级转化为tokens

```python
def fit_text_to_sequences(self, texts):
    """
    将所有句子转化为token
    :param texts:list of str,no separate in str, shape: list[str]
    比如:["飘渺孤鸿影，漏断人初静",”想当年，金戈铁马，气吞万里如虎“]
    :return:token_list:list of tokens,shape list[tokens]
    """
    if not isinstance(texts, list) or not isinstance(texts[0], str):
        raise Exception("texts must be a list[str] type")

    if not self.__char_level:
        new_texts = self.text_to_wordlevel(texts)
        return self.__TOKENIZER.texts_to_sequences(new_texts)
    else:
        return self.__TOKENIZER.texts_to_sequences(texts)
```
## 数据处理（词向量）

在使用词向量的情况下，分词仍然是使用jieba分词，然后使用Data_wdvec.py这个文件，文件中的函数如下：

创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引

```python
def word2vec_train(text_sep):
    model = Word2Vec(vector_size=vocab_dim,  # 特征向量维度
                     min_count=n_exposures,  # 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
                     window=window_size,  # 窗口大小，表示当前词与预测词在一个句子中的最大距离是多少
                     workers=cpu_count,  # 用于控制训练的并行数
                     )
    model.build_vocab(text_sep)  # 创建词汇表， 用来将 string token 转成 index
    model.train(text_sep, total_examples=model.corpus_count, epochs=10)
    model.save('Word2vec_model.pkl')  # 保存训练好的模型
    word2idx, word2vec, text_sep = create_dictionaries(model=model, text_sep=text_sep)
    return word2idx, word2vec, text_sep  # word_vectors字典类型{word:vec}
```
```python
def create_dictionaries(model=None, text_sep=None):
    if (text_sep is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model.wv[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def ToIdx(texts):
            data = []
            for sentence in texts:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        text_sep = ToIdx(text_sep)
        text_sep = sequence.pad_sequences(text_sep, maxlen=truncate_thereshold)  # 前方补0 为了进入LSTM的长度统一
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, text_sep
    else:
        print('No data provided...')
```

构建词向量矩阵
```python
def get_data(word2idx, word2vec):
    n_symbols = len(word2idx) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in word2idx.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word2vec[word]

    return n_symbols, embedding_weights
```

## 搭建模型

- Embedding层
  - 词嵌入，将index转化为vector
- Bi-LSTM层
- Dense层
  - 通过softmax激活，映射到多分类问题

```python
def define_lstm_model(word2token_len):
    model = Sequential()
    # input_shape是输入的维度特征,output_dim是输出词向量的维度，是一个随机初始化的过程
    model.add(Embedding(output_dim=output_dim,input_dim=word2token_len,mask_zero=True))
    model.add(Bidirectional(LSTM(LSTM_UNITS)))
    model.add(Dense(3,activation="softmax"))
    model.compile(optimizer='adam',metrics=["accuracy"],loss="sparse_categorical_crossentropy")
    return model
```

以及

- Embedding层

  - 词嵌入，将index转化为vector

- Conv1D层


  - ```python
    Conv1D(filters=filters,kernel_size=3,activation='relu',strides=1)
    ```

- BiLSTM层

- Dense层

```python
def define_word_avg_model(word2token,truncate_thereshold):
    # input_shape是输入的维度特征,output_dim是输出词向量的维度，是一个随机初始化的过程
    model = Sequential()
    model.add(Embedding(output_dim=output_dim,input_dim=len(word2token),input_length=truncate_thereshold,mask_zero=True))
    # 这里需要些模型
    # filters是输出的向量长
    model.add(Conv1D(filters=filters,kernel_size=3,activation='relu',strides=1))
    model.add(Bidirectional(LSTM(LSTM_UNITS)))
    model.add(Dense(3,activation="softmax"))
    model.compile(optimizer='adam',metrics=["accuracy"],loss="sparse_categorical_crossentropy")
    return model

```
若使用词向量，则使用单向LSTM，嵌入层使用词向量矩阵初始化
```python
def define_lstm_model(n_symbols, embedding_weights, truncate_thereshold):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever #堆叠
    # 嵌入层将正整数（下标）转换为具有固定大小的向量
    model.add(Embedding(output_dim=output_dim,  # 词向量的维度
                        input_dim=n_symbols,  # 字典(词汇表)长度
                        mask_zero=True,  # 确定是否将输入中的‘0’看作是应该被忽略的‘填充’（padding）值
                        weights=[embedding_weights], # 词向量矩阵
                        input_length=truncate_thereshold))
    model.add(LSTM(units=50, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 全连接层
    model.add(Activation('softmax'))

    print('Compiling the Model...')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
```


## 训练模型

以word_lstm为例，首先定义checkpoint，以验证集的损失为单位，保存

```python
# 定义一个CheckPoint
CheckPoint = ModelCheckpoint(save_best_only=True,monitor="val_loss",filepath="model/word_lstm")
```

然后训练，并且保存历史，以供后续制图

```python
history = model.fit(pad_seqs,labels,epochs=EPOCHES,batch_size=32,callbacks=[CheckPoint],class_weight=class_weights,validation_split=.1)
    model.save("model/word_lstm_model.h5")

    with open("model/word_lstm_history.txt",mode='wb') as f:
        pickle.dump(history.history,f)
```

## 超参数说明

```python
EPOCHES = 30
output_dim = 30
LSTM_UNITS = 30
BATCH_SIZE = 64
filters = 30
kernel_size = 3
strides = 1
```

## 实验结果和说明



## 总结

一开始做实验的时候，是因为以字符形式输入到嵌入层，和中文语义中的分词是有所区别，而且违反直觉的。而且在英文中，却是分英文单词，而不是分英文字符的。在课程设计中，我们决定做一个实验去探索这一点。由于词相比字的稀疏性，在本数据集40000条左右的情况下，可以看到，中文分词输入由于词比字更稀疏，过拟合非常的明显，以非常少的词就可以判断训练集的分类。但这些词对验证集来说，可能一点影响没有，或者验证集中，干脆找不到这些词，这些词可能出现的频率就很低。

在不断地学习过程中，词输入的BiLSTM层，更进一步的，学习了无关紧要的信息，进一步精细化的减少训练集的损失，这也使得学习到的词越来越对分类有决定性作用，可能就是生僻词越来越多，这也使得测试集的loss在不断地增加，acc出现了下降后的震荡。

一个可能的解决方法，是先对训练集的词做一个筛选，只留下高频词，这会提升以词为输入的训练模型的准确度。

以词为输入的训练模型，在训练集上的表现比字输入要更好，但是过拟合的情况也特别明显，需要做额外的步骤处理，泛化。

对于大部分情况来说，以字为输入虽然收敛要慢一些，但是过拟合的情况，即使发生，也可以通过正则化或者Dropout等手段调节。而对词来说，这种严重的过拟合趋势，必须要对输入做一些处理。
