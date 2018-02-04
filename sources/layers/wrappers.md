<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L104)</span>
### TimeDistributed

```python
keras.layers.TimeDistributed(layer)
```
这个包装器将一个图层应用于输入的每个时间片。

输入应该至少是3D，而且索引1的维度将被视为一个时间维度。

考虑一批三十二个样本, 其中每个样本是16个维度的10个向量的序列。
该层的批量输入形状是`(32，10，16）`，
而不包括样本维度的`input_shape`是`（10,16）`。

然后，您可以使用`TimeDistributed`来应用`Dense`图层到10个时间步的每一个，独立:

```python
# as the first layer in a model
model = Sequential()
model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)
```

输出将有形状`（32，10，8）`。

在后续的图层中，不需要`input_shape`:

```python
model.add(TimeDistributed(Dense(32)))
# now model.output_shape == (None, 10, 32)
```

输出将具有形状`（32，10，32）`。

`TimeDistributed`可以用于任意图层，而不仅仅是`Dense`,
例如一个`Conv2D`层：

```python
model = Sequential()
model.add(TimeDistributed(Conv2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__参数__

- __layer__: 一个图层实例。

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/wrappers.py#L221)</span>
### Bidirectional

```python
keras.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

RNN的双向包装器。

__参数__

- __layer__: `Recurrent`实例。
- __merge_mode__: 前向和后向RNN如何合并输出模式。
以下之一：`{'sum', 'mul', 'concat', 'ave', None}`。
如果`None`，输出将不会合并，它们将作为列表返回。

__引发__

- __ValueError__: 如果无效的`merge_mode`参数。

__例子__


```python
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```
