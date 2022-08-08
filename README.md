
### Invisible MegEngine

Goal: bridging the gap between megengine and torch.

只需要把所有torch相关的code转换成invis就行了

比如：

```python
import invis as torch
import invis.nn.functional as F
```

Enjoy!

#### Installation

```shell
git clone git@github.com:MegEngine/invis.git
cd invis
```

个人建议使用[venv](https://docs.python.org/3/library/venv.html)，如果不想用，可以跳过这一步

```shell
python3 -m venv invis_venv ~/invis_venv
source ~/invis_venv/bin/activate
```

使用pip进行安装

```shell
pip3 install -r requirements.txt
pip3 install -v -e .
```

#### Features

* invis.nn.Moudle不会把builtin 的 dict 和 list 看做ModuleDict和 ModuleList了，你可以放心的往Module里塞入Tenor和Module而不用担心state_dict里面多出来一些奇怪的东西了
* 每个Module终于可以自定义load_state_dict的逻辑了
* 诸如 x.abs().sigmoid() 终于可以用了，抛弃掉诸如F.sigmoid(F.abs(x))的调用吧
* 增加了一些方法，诸如meshgrid、where、pixel_shuffle等
* 修复mge里面一些支持不全的功能，比如pad(x, (-2, -2, -2, -2))这种

#### Examples

作为对invis的磨练，我转换了一些基本的模型。

* 大部分模型来自于直接对torchvision的转换，使用的版本为0.12.0
* 检测部分写了YOLOX是因为相对来说YOLOX的写法还是有一部分的复杂性的(而且自己也很熟)
* realcu-gan纯粹是个人兴趣所在(谁不想看到高清的老番呢)

##### 分类模型

* [alexnet](./examples/alexnet)
* [densenet](./examples/densenet)
* [googlenet](./examples/googlenet)
* [inception](./examples/inception)
* [resnet](./examples/resnet)
* [shufflenet](./examples/shufflenet)
* [squeezenet](./examples/squeezenet)
* [SwinTransformer](./examples/swin_transformer)

##### 检测模型

* [YOLOX](./examples/yolox)

##### 分割网络

* [FCN](./examples/fcn)

##### GAN

* [realcu-GAN](./examples/realcu-gan)

#### Why invis ?

invis的初衷还是为了减少复杂度(complexity)，准确来说是R可控的复杂度。我已经厌倦了告诉别人：
* mge BatchNorm的momentum和torch的不一样，如果torch的是0.9，那么mge的是0.1
* 转权重的时候，group conv 和 torch 也是不一样的，bias也不一样。
* Linear的初始化也不太一样，之前我们复现DETR也因为这个差了一些点。

现在，我只需要告诉他，这个坑在invis里面有，你可以去看一下。

除此之外，还有一些其他可能的使用场景：

* 对inference结果
* 将torch代码尽量快地转成trace module
* 需要一个torch训练好的backbone来做预训练，但是懒得转weight

**invis的用户有多少，并不取决于我的算子包的多好，而是用户将代码从torch切到megengine之后，能享受到什么好处。**

#### Contribution

invis仅仅在一个小范围内进行了打磨，而且一些corner case未必支持地完全，任何MR/PR和issue都是欢迎的。
