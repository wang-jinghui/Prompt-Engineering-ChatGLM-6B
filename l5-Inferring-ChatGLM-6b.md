# 推断

&emsp;&emsp;推断可以看作是模型接受文本作为输入并进行某种分析的任务。因此，这可能涉及标签提取、内容理解和情感分析等。如果你想要从一段文本中提取情感，无论是积极的还是消极的，在传统的机器学习工作流程中，你需要收集标签数据集、训练模型、然后部署模型并进行推断。这样做可能效果不错，但需要完成很多繁琐的工作。而且对于每个任务，你都需要训练并部署单独的模型。

&emsp;&emsp;大语言模型的优势是，对于许多这样的任务，你只需要编写提示即可开始生成结果。极大地提高了应用开发的速度。而且你只需要使用一个模型、一个API来完成许多不同的任务，而且不需要训练和部署许多不同的模型。

### 一：环境设置

加载开源的**chatGLM**模型,使用ChatGLM-6b的INT8版本。


```python
import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModel
```


```python
tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b", trust_remote_code=True).half().quantize(8).to('cuda')
```

    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.
    Explicitly passing a `revision` is encouraged when loading a configuration with custom code to ensure no malicious code has been contributed in a newer revision.
    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.



    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]



```python
def get_completion(prompt, history=[], temperature=0.95):
    
    response, _ = model.chat(tokenizer=tokenizer, query=prompt, history=history,
                                   temperature=temperature)
    return response
```

### 任务1：电商评价的情感分类


```python
lamp_review_zh = """
我需要一盏漂亮的卧室灯，这款灯具有额外的储物功能，价格也不算太高。\
我很快就收到了它。在运输过程中，我们的灯绳断了，但是公司很乐意寄送了一个新的。\
几天后就收到了。这款灯很容易组装。我发现少了一个零件，于是联系了他们的客服，他们很快就给我寄来了缺失的零件！\
在我看来，Lumina 是一家非常关心顾客和产品的优秀公司！
"""
```


```python
prompt = f"""
以下用三个反引号分隔的产品评论的情感是什么？

评论文本: ```{lamp_review_zh}```
"""
response = get_completion(prompt)
print(response)
```

    这段评论的情感是积极的。评论中提到了这款卧室灯具有额外的储物功能,价格不算太高,而且公司在运输过程中提供了新的灯绳,并且在发现缺少零件后,很快地提供了缺失的零件。这些表现表明评论者对这款灯的评价很高,认为它是一个很好的产品。因此,整个评论的情感是积极的。


更精简的回复：积极/消极


```python
prompt = f"""
以下用三个反引号分隔的产品评论的情感是什么？

用一个词回答：「积极」或「消极」。

评论文本: ```{lamp_review_zh}```
"""
response = get_completion(prompt)
print(response)
```

    积极


### 任务2：愤怒识别

识别用户评论中是否包含愤怒的情绪。


```python
prompt = f"""
识别下面用```包括的评论中作者是否表达了愤怒的情绪。

用一个词回答：是 或 否。

评论文本: ```{lamp_review_zh}```
"""
response = get_completion(prompt)
print(response)
```

    否。


### 任务3：信息提取并格式化

&emsp;&emsp;通过LLM做的更多事情，特别是从客户评论中获取更丰富的信息。信息抽取是自然语言处理的一部分，在下面提示中，要求模型执行一系列的任务：1. 识别购买的商品和制造商品的公司名称。 2.情感分析和愤怒检测。3.格式化输出为json对象。
 


```python
prompt = f"""
从评论文本中识别以下内容，评论用三个反引号分隔:
- 情绪（积极或消极）
- 是否表达了愤怒情绪？（是或否）
- 评论者购买的物品名
- 制造该物品的公司名

把识别结果保存为一个Json对象，\
"Sentiment", "Anger", "Item" and "Brand" 作为key.

评论文本: ```{lamp_review_zh}```
"""
response = get_completion(prompt)
print(response)
```

    ```json
    {
      "Sentiment": "积极",
      "Anger": false,
      "Item": "卧室灯",
      "Brand": "Lumina"
    }
    ```


### 任务4：主题推断

给定一段长文本，推断这段文本是关于什么主题。


```python
story_zh = """
在政府最近进行的一项调查中，要求公共部门的员工对他们所在部门的满意度进行评分。\
调查结果显示，NASA是最受欢迎的部门，满意度为95％。\
一位NASA员工John Smith对这一发现发表了评论，他表示：\
“我对NASA排名第一并不感到惊讶。这是一个与了不起的人们和令人难以置信的机会共事的好地方。我为成为这样一个创新组织的一员感到自豪。”\
NASA 的管理团队也对这一结果表示欢迎，主管Tom Johnson表示：\
“我们很高兴听到我们的员工对NASA的工作感到满意。\
我们拥有一支才华横溢、忠诚敬业的团队，他们为实现我们的目标不懈努力，看到他们的辛勤工作得到回报是太棒了。”\
调查还显示，社会保障管理局的满意度最低，只有45％的员工表示他们对工作满意。\
政府承诺解决调查中员工提出的问题，并努力提高所有部门的工作满意度。"""
```


```python
prompt = f"""
确定以下给定文本中讨论的五个主题。

每个主题用不超过二十个字。

输出时用逗号分割每个主题。

给定文本: ```{story_zh}```
"""
response = get_completion(prompt)
print(response)
```

    1. 调查结果显示,NASA是最受欢迎的部门,满意度为95%。
    2. 一位NASA员工John Smith对NASA排名第一并不感到惊讶。
    3.NASA的管理团队也对这一结果表示欢迎。
    4. 调查还显示,社会保障管理局的满意度最低,只有45%的员工表示他们对工作满意。
    5. 政府承诺解决调查中员工提出的问题,并努力提高所有部门的工作满意度。


### 任务5：为特定的主题制作提醒

&emsp;&emsp;通过LLM检测文本中的主题，只通过提示进行主题推断，这在机器学习中，被称为zero-shot学习算法，因为我们没有给它任何已标记的训练数据。所以这是zero-shot。使用此技术，您可以快速并准确地确定新闻文章中所涉及的主题，然后更好地理解文章的主旨和内容。


```python
prompt = f"""
判断给定的文本中是否包含主题列表中的每一个主题，

如果包含用1表示，否则0。

给定的文本用```符号分隔，

主题列表：[NASA，部门满意度，联邦政府，地方政府]。 

给定文本: ```{story_zh}```
"""
response = get_completion(prompt)
print(response)
```

    这段文本包含主题列表中的三个主题:NASA、部门满意度、联邦政府。
    
    因此,该文本的值为3,即包含这三个主题。


### 测试


```python
text = f"""
昨天晚上每人发了一个苹果，
我们年轻人的苹果上都是虫眼，老干部手里的苹果都溜光水滑。

年轻人：“这苹果都生虫了，怎么吃！”

老干部：“虫子都不敢吃的东西，你敢吃啊？”

年轻人：“那你们的苹果都没虫眼啊！”

老干部：“我们都多大岁数了，哪还在乎那么多！”
"""

prompt = f"""
给定文本中的情感是什么，

用不超过二十个字概括，

给定文本: ```{text}```
"""
response = get_completion(prompt)
print(response)
```

    文本中的情感是幽默和调侃。



```python
text = f"""
1985年，只生一个好，政府来养老。
1995年，只生一个好，政府帮养老。
2005年，养老不能靠政府。
2012年，推迟退休好，自己来养老。
2013年，以房养老。
2018年，赡养老人是义务，推给政府很可耻。
2020年：养儿为防老，子女要尽孝，甩给政府管，真是脸不要。
2023年：子女是耐消品，买个来养老。
"""

prompt = f"""
给定文本中的主题和情感分别是什么，

用不超过二十个字概括，

按以下格式输出结果：
- 主题：
- 情感：

给定文本: ```{text}```
"""
response = get_completion(prompt)
print(response)
```

    主题:政府养老观念逐渐淡化,个人养老逐渐流行。
    情感:负面,对政府养老的不信任和对个人养老的自信形成对比。

