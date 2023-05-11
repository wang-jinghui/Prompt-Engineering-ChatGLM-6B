## 提示指南


### 一、 环境配置

加载开源的**chatGLM**模型,使用ChatGLM-6b的INT8版本。


```python
import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings('ignore')
```


```python
tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b", trust_remote_code=True).half().quantize(8).to('cuda')
```

    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.
    Explicitly passing a `revision` is encouraged when loading a configuration with custom code to ensure no malicious code has been contributed in a newer revision.
    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.



    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]


**辅助函数**：
输入提示，返回生成结果和历史


```python
def get_completion(prompt, history, temperature=0.9):
    
    response, history = model.chat(tokenizer=tokenizer, query=prompt, history=history,
                                   temperature=temperature)
    return response, history
```

## 二、两个基本原则

### 原则一：编写清晰、具体的指令

你应该通过提供尽可能清晰和具体的指令来表达您希望模型执行的操作。这将引导模型给出正确的输出，并减少你得到无关或不正确响应的可能。编写清晰的指令不意味着简短的指令，因为在许多情况下，更长的提示实际上更清晰且提供了更多上下文，这实际上可能导致更详细更相关的输出。


#### 策略一：使用分隔符表示输入的不同部分

分隔符可以是任一种：` ```，""，<>，\<tag>，<\tag>`，能够使模型明确知道这是一个独立部分。使用定界符也是一种有用的技术，可以避免提示注入。提示注入，是指如果允许用户向提示中添加一些输入，它们可能会向模型提供一些冲突的指令，从而使模型遵循错误的指令而不是执行你所期望的操作。

 



```python
# 需要总结的文本内容部分
text = f"""
您应该通过提供尽可能清晰和具体的说明来表达您希望模型执行的操作。\
这将引导模型达到所需的输出，并减少收到不相关或不正确响应的机会。\
不要将编写清晰的提示与编写简短的提示混淆。在许多情况下，\
较长的提示为模型提供了更清晰的上下文，这可以导致更详细和相关输出。
"""
# 指令部分
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    By providing clear and detailed instructions for the actions you want the model to perform, you can guide the model towards the desired output and reduce the chance of receiving relevant or incorrect responses. It's important to avoid混淆 between clear and简短的 instructions. In many cases, longer instructions provide a更清晰 context for the model, leading to more detailed and relevant output.


下面简单修改一下prompt,加上`use chinese`


```python
text = f"""
您应该通过提供尽可能清晰和具体的说明来表达您希望模型执行的操作。\
这将引导模型达到所需的输出，并减少收到不相关或不正确响应的机会。\
不要将编写清晰的提示与编写简短的提示混淆。在许多情况下，\
较长的提示为模型提供了更清晰的上下文，这可以导致更详细和相关输出。
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence use chinese.
```{text}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    表述清晰和具体,以引导模型执行所需的操作,并减少收到不相关或不正确响应的机会。提供尽可能清晰的提示,但不要将简短的提示混淆为清晰的提示。较长的提示可以为模型提供更清晰的上下文,导致更详细和相关的输出。



```python
text = f"""
您应该通过提供尽可能清晰和具体的说明来表达您希望模型执行的操作。\
这将引导模型达到所需的输出，并减少收到不相关或不正确响应的机会。\
不要将编写清晰的提示与编写简短的提示混淆。在许多情况下，\
较长的提示为模型提供了更清晰的上下文，这可以导致更详细和相关输出。
"""
prompt = f"""
用一句话总结由三反引号分隔的文本。
```{text}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    使用清晰和具体的说明来引导模型执行操作,避免收到不相关或不正确响应,同时不要将编写清晰的提示与编写简短的提示混淆,较长的提示可以为模型提供更清晰的上下文,导致更详细和相关的输出。



```python
text = f"""
You should express what you want a model to do by \ 
providing instructions that are as clear and \ 
specific as you can possibly make them. \ 
This will guide the model towards the desired output, \ 
and reduce the chances of receiving irrelevant \ 
or incorrect responses. Don't confuse writing a \ 
clear prompt with writing a short prompt. \ 
In many cases, longer prompts provide more clarity \ 
and context for the model, which can lead to \ 
more detailed and relevant outputs.
"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
```{text}```
"""

response, history = get_completion(prompt, [])
print(response)
```

    Providing clear and specific instructions for a model can help reduce the chances of its receiving relevant or incorrect responses. It's important to avoid confusion about the desired output and to lengthen prompts to provide more context for the model.


#### 策略2：结构化模型的输出


```python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre, use chinese.
"""
response, hisotry = get_completion(prompt, history)
print(response)
```

    Here's a list of three made-up book titles in JSON format with their authors and genres:
    
    ```json
    {
      "book_id": 1,
      "title": "我的奇妙冒险",
      "author": "小明",
      " genre": "奇幻"
    }
    
    {
      "book_id": 2,
      "title": "神秘岛",
      "author": "小红",
      " genre": "科幻"
    }
    
    {
      "book_id": 3,
      "title": "哈利波特与魔法石",
      "author": "哈利波特",
      " genre": "魔法奇幻"
    }
    ```
    
    Note: The use of the Chinese characters "我的奇妙冒险" (mǔ dào qiǎo lì) and "神秘岛" (shén huī qián dào) represents the fictional book titles that I made up.


#### 策略 3: 要求模型检查条件是否满足

如果任务有假设条件并且这些条件不一定被满足，那么我们可以要求模型首先检查这些假设条件，如果不满足则指出来，并停止任务。如果满足执行任务。现在我将复制一段描述如何泡茶的段落到提示中。提示是，如果文本包含一系列指示，请将这些指示重写为以下格式，然后写出步骤说明。如果文本不包含一系列指示，则只需写下“未提供步骤”。


```python
text_1 = f"""
Making a cup of tea is easy! First, you need to get some \ 
water boiling. While that's happening, \ 
grab a cup and put a tea bag in it. Once the water is \ 
hot enough, just pour it over the tea bag. \ 
Let it sit for a bit so the tea can steep. After a \ 
few minutes, take out the tea bag. If you \ 
like, you can add some sugar or milk to taste. \ 
And that's it! You've got yourself a delicious \ 
cup of tea to enjoy.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
response, hisotry = get_completion(prompt, history)
print("Completion for Text 1:")
print(response)
```

    Completion for Text 1:
    No steps provided.



```python
text_1 = f"""
泡一杯茶很容易。首先，需要把水烧开。\
在等待期间，拿一个杯子并把茶包放进去。\
一旦水足够热，就把它倒在茶包上。\
等待一会儿，让茶叶浸泡。几分钟后，取出茶包。\
如果你愿意，可以加一些糖或牛奶调味。\
就这样，你可以享受一杯美味的茶了。
"""
prompt = f"""
您将获得由三个引号括起来的文本。\
如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

第一步 - ...
第二步 - …
…
第N步 - …

如果不包含一系列的指令，则直接写“未提供步骤”。"
\"\"\"{text_1}\"\"\"
"""
response, hisotry = get_completion(prompt, [])
print("Completion for Text 1:")
print(response)
```

    Completion for Text 1:
    第一步: 将水烧开
    第二步: 把茶包放入杯子中
    第三步: 把水倒在茶包上
    第四步: 等待水足够热
    第五步: 把茶包取出
    第六步: 让茶叶浸泡
    第七步: 等待几分钟后
    第八步: 取出茶包
    第九步: 加一些糖或牛奶
    第十步: 调味
    
    未提供步骤



```python
text_2 = f"""
The sun is shining brightly today, and the birds are \
singing. It's a beautiful day to go for a \ 
walk in the park. The flowers are blooming, and the \ 
trees are swaying gently in the breeze. People \ 
are out and about, enjoying the lovely weather. \ 
Some are having picnics, while others are playing \ 
games or simply relaxing on the grass. It's a \ 
perfect day to spend time outdoors and appreciate the \ 
beauty of nature.
"""
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_2}\"\"\"
"""
response, hisotry = get_completion(prompt, history)
print("Completion for Text 2:")
print(response)
```

    Completion for Text 2:
    No steps provided.



```python
text_2 = f"""
今天阳光明媚，鸟儿在歌唱。\
这是一个去公园散步的美好日子。\
鲜花盛开，树枝在微风中轻轻摇曳。\
人们外出享受着这美好的天气，有些人在野餐，有些人在玩游戏或者在草地上放松。\
这是一个完美的日子，可以在户外度过并欣赏大自然的美景。
"""
prompt = f"""
您将获得由三个引号括起来的文本。\
如果它包含一系列的指令，则需要按照以下格式重新编写这些指令：

第一步 - ...
第二步 - …
…
第N步 - …

如果文本中不包含一系列的指令，则直接写“未提供步骤”。"
\"\"\"{text_2}\"\"\"
"""
response, hisotry = get_completion(prompt, history, temperature=0.99)
print(response)
```

    第一步: 阳光明媚
    第二步: 鸟儿在歌唱
    第三步: 去公园散步
    第四步: 鲜花盛开
    第五步: 树枝在微风中轻轻摇曳
    第六步: 人们外出享受着这美好的天气
    第七步: 有些人在野餐,有些人在玩游戏或者在草地上放松
    第八步: 这是一个完美的日子,可以在户外度过并欣赏大自然的美景
    第九步: 结束。
    
    未提供步骤。


#### 策略 4: 少量提示

告诉模型它的任务是以一致的风格回答问题，我们提供了一个孩子和祖父之间的对话示例。孩子说：“教我耐心”，祖父用类比的方式回答。既然我们要求模型用一致的语气回答。


```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest valley flows from a modest spring; the \ 
grandest symphony originates from a single note; the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
response, hisotry = get_completion(prompt, [])
print(response)
```

    <Grandparent>: Resilience is the ability to withstand and overcome挫折 and adversity. It is the key to building up a strong character and standing up to the challenges that life throws at us. It is the ability to come back from a difficult situation and carry on with your life in a positive way.



```python
prompt = f"""
你的任务是以一致的风格回答问题。

<孩子>: 教我耐心。

<祖父母>: 挖出最深峡谷的河流源于一处不起眼的泉眼；最宏伟的交响乐从单一的音符开始；最复杂的挂毯以一根孤独的线开始编织。

<孩子>: 教我韧性。
"""
response, hisotry = get_completion(prompt, [])
print(response)
```

    韧性是一种品质,指物体在受到外力作用时不易断裂或变形的能力。在生活中,韧性可以帮助我们更好地应对挑战和压力。以下是一些提高韧性的方法:
    
    1. 坚持锻炼:锻炼可以增强身体的韧性,使我们更加坚韧和灵活。
    
    2. 学会放松:过度紧张和焦虑会影响韧性,因此学会放松身心可以帮助我们更好地控制自己的情绪和思维。
    
    3. 学会规划:有目的地安排时间和任务可以提高我们的韧性,因为我们可以更好地控制自己的行为和情绪。
    
    4. 培养耐心:耐心是一种重要的品质,可以帮助我们更好地处理困难和挑战。
    
    5. 学会包容:包容他人的不同观点和想法可以帮助我们更好地控制自己的情绪和态度,从而提高我们的韧性。


### 原则2-给模型充足的思考时间

给模型充足的思考时间,如果模型由于急于得出错误的结论而出现了推理错误，您可以尝试重新构造查询，要求模型在提供最终答案之前进行一系列相关推理。另一种思考方式是，如果您给模型一个时间太短或用太少的字数来完成的任务，它可能会猜测答案，这个答案很可能是错误的。你知道，这对一个人来说也一样。

#### 策略1：指定完成任务所需的具体步骤

下面的文本是描述杰克和吉尔（Jack and Jill）故事的段落。在提示中，指令是执行一系列的动作：
- 第一，用一句话总结由三个反引号包围的文本。
- 第二，将摘要翻译成法语。
- 第三，列出法语摘要中的每个名字。
- 第四，输出一个JSON对象，包含以下键：chinese_summary和num_names。
- 最后，我们希望用换行符分隔答案。 


```python
text = f"""
在一个迷人的村庄里，兄妹杰克和吉尔出发去一个山顶井里打水。\
他们一边唱着欢乐的歌，一边往上爬，\
然而不幸降临——杰克绊了一块石头，从山上滚了下来，吉尔紧随其后。\
虽然略有些摔伤，但他们还是回到了温馨的家中。\
尽管出了这样的意外，他们的冒险精神依然没有减弱，继续充满愉悦地探索。
"""
# example 1
prompt = f"""
执行以下操作：
1-用一句话概括下面用三个反引号括起来的文本。
2-将摘要翻译成英语。
3-在英语语摘要中列出每个人名。
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。

请用换行符分隔您的答案。

Text:
```{text}```
"""

response, hisotry = get_completion(prompt, [])
print("Completion for prompt:")
print(response)
```

    Completion for prompt:
    1. 概括:描述兄妹去山顶井里打水的故事,其中杰克意外滚下山,吉尔紧随其后摔伤,但他们依然充满冒险精神继续探索。
    2. 翻译:
    
       在一个迷人的村庄里,兄妹杰克和吉尔出发去一个山顶井里打水。他们一边唱着欢乐的歌,一边往上爬,然而不幸降临——杰克绊了一块石头,从山上滚了下来,吉尔紧随其后。虽然略有些摔伤,但他们还是回到了温馨的家中。尽管出了这样的意外,他们的冒险精神依然没有减弱,继续充满愉悦地探索。
    3. 列出每个人名:
    
       杰克,吉尔,小村庄里的其他人。
    4. 输出 JSON 对象:
    
       ```
       {
         "English_summary": "兄妹去山顶井里打水的故事,其中杰克意外滚下山,吉尔紧随其后摔伤,但他们依然充满冒险精神继续探索。",
         "num_names": 3
       }
       ```


- 概括：&#10004;
- 翻译：&#10008;
- 人名：&#10008;
- 格式化：&#10004;

**中英文提示效果对比**：


```python
prompt = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into Chinese.
3 - List each name in the Chinese summary.
4 - Output a json object that contains the 
  following keys: Chinese_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
response, hisotry = get_completion(prompt, history)
print("\nCompletion for prompt 2:")
print(response)
```

    
    Completion for prompt 2:
    Summary: 
    
    In an迷人的村庄,杰克和吉尔出发去山顶井里打水,但他们在攀爬过程中遭遇不幸,杰克绊了一块石头并从山上滚了下来,吉尔紧随其后。虽然他们略有些摔伤,但他们还是回到了家中。尽管出了这样的意外,他们的冒险精神依然没有减弱,继续充满愉悦地探索。
    
    Translation:
    
    In a beautiful village, Jack and吉尔 went to the top of a mountain well to fetch water, but they got into an accident while climbing. Jack fell from the mountain and got injured, but吉尔 followed suit. Although they were slightly injured, they returned home feeling brave and excited to explore more.
    
    Names: 
    
    In the summary, there are two names mentioned,杰克和吉尔. The list of names in the Italian summary does not include these two names.
    
    Output JSON: 
    
    ```
    {
      "Chinese_summary": "杰克和吉尔去山顶井里打水,但他们在攀爬过程中遭遇不幸,杰克绊了一块石头并从山上滚了下来,吉尔紧随其后。虽然他们略有些摔伤,但他们还是回到了家中。尽管出了这样的意外,他们的冒险精神依然没有减弱,继续充满愉悦地探索。",
      "num_names": 2
    }
    ```


-----------------


```python
prompt = f"""
1-用一句话概括下面用<>括起来的文本。
2-将摘要翻译成英语。
3-在英语摘要中列出每个名称。
4-输出一个 JSON 对象，其中包含以下键：English_summary，num_names。

请使用以下格式：
文本：<要总结的文本>
摘要：<摘要>
翻译：<摘要的翻译>
名称：<英语摘要中的名称列表>
输出 JSON：<带有 English_summary 和 num_names 的 JSON>

Text: <{text}>
"""
response, hisotry = get_completion(prompt, [])
print("\nCompletion for prompt :")
print(response)
```

    
    Completion for prompt :
    摘要:
    
    在迷人的村庄里,兄妹杰克和吉尔出发去山顶井里打水,但杰克绊了一块石头并从山上滚了下来,吉尔紧随其后。虽然略有些摔伤,但他们还是回到了家中。尽管出了这样的意外,他们的冒险精神依然没有减弱,继续充满愉悦地探索。
    
    翻译:
    
    On a charming village,兄妹杰克 and吉尔 set out to draw water from a mountain well. They sing happy songs as they climb, but unfortunately, something goes wrong -杰克 falls from the mountain and摔了下来, while吉尔 follows him. Although they are slightly injured, they return home safely. Although they faced such an unexpected accident, their adventure spirit did not减弱, and they continued to explore with joy.
    
    名称:
    
    The names in the English summary are: 杰克, 吉尔, and the mountain.


对于chatGLM-6B来说，英文指令效果更好一些。

#### 策略 2: 指导模型(在急于得出结论之前)制定自己的解决方案

有时候，我们明确指导模型在做出结论之前自行推理出解决方案，会得到更好的结果，这和之前提到的给模型时间思考问题，不要急于下结论的概念相同。因此，在这个问题中，我们要求模型确定学生的解决方案是否正确。


```python
prompt = f"""
判断学生的解决方案是否正确。

问题:
我正在建造一个太阳能发电站，需要帮助计算建设费用。

    土地价格为100美元/平方英尺。
    太阳能电池板的价格为250美元/平方英尺。
    维护费用为每年支付固定的100000美元，另外支付10美元/平方英尺。
    现假设发电站的面积为x平方英尺，首年运营的总费用是多少。

学生的解决方案：
设x为发电站的大小，单位为平方英尺。
费用：

    土地费用：100x
    太阳能电池板费用：250x
    维护费用：100000+100x
    总费用：100x+250x+100000+100x=450x+100000
"""
response, hisotry = get_completion(prompt, [], temperature=0.9)
print(response)
```

    学生的解决方案是正确的。根据题目中提供的信息,我们可以列出所有相关的费用,然后将它们相加来计算总费用。土地价格为100美元/平方英尺,因此x乘以100等于土地费用;太阳能电池板价格为250美元/平方英尺,因此250x等于太阳能电池板费用;维护费用为每年支付固定的100000美元,另外支付10美元/平方英尺,因此100000+10x等于维护费用。将这些费用相加,得到:
    
    100x + 250x + 100000 + 10x = 450x + 100000
    
    将等式两边都除以450,得到:
    
    x = 92.3
    
    因此,发电站的最大面积为92.3平方英尺。


&emsp;&emsp;学生的答案是错的，维护费用应该是：100000+10x，总费用：360x+100000。为了解决这个问题，我们需要让模型自己解决问题，然后将自己的解决方案与学生的解决方案进行比较，评估学生的解决方案是否正确。在此之前，不要判断学生的解决方案是否正确，一定要确保自己已经清晰地理解了这个问题。


```python
prompt = f"""
请判断学生的解决方案是否正确，请通过如下步骤解决这个问题：

步骤：

    首先，自己计算一下首年的费用，仔细计算维护的费用。
    然后将你的解决方案与学生的解决方案进行比较，然后判断学生的解决方案是否正确。
    在自己完成问题之前，请勿决定学生的解决方案是否正确。

使用以下格式：

    问题：
    学生的解决方案：学生的解决方案文本
    实际解决方案和步骤：实际解决方案和步骤文本
    学生的解决方案和实际解决方案是否相同：是或否
    学生的成绩：正确或不正确

问题:
我正在建造一个太阳能发电站，需要帮助计算财务。

    土地价格为100美元/平方英尺。
    太阳能电池板的价格为250美元/平方英尺。
    维护费用为每年支付固定的100000美元，
    另外维护单价为10美元/平方英尺。
    现假设发电站的面积为x平方英尺，首年运营的总费用是多少。

学生的解决方案：
太阳能发电站的大小为x平方英尺。
费用：

    土地费用：100x
    太阳能电池板费用：250x
    维护费用：100000+100x
    总费用：100x+250x+100000+100x=450x+100,000

实际解决方案和步骤：
"""
response, hisotry = get_completion(prompt, [], temperature=0.95)
print(response)
```

    实际解决方案和步骤:
    
    由于学生的解决方案中没有考虑太阳能电池板的使用寿命和维护费用,因此他的解决方案是错误的。
    
    实际解决方案和步骤:
    
    1. 计算土地费用和太阳能电池板费用。根据题目中提供的信息,土地价格为100美元/平方英尺,太阳能电池板的价格为250美元/平方英尺。因此,土地费用为100x,太阳能电池板费用为250x。
    
    2. 计算每年的维护费用。维护费用为每年支付固定的100000美元,并且维护单价为10美元/平方英尺。因此,每年的维护费用为100000+10x。
    
    3. 计算首年运营的总费用。由于发电站的面积为x平方英尺,因此首年运营总费用为100x+250x+100000+10x=450x+100,000美元。
    
    4. 比较实际解决方案和步骤,并判断学生的解决方案是否正确。由于学生的解决方案中没有考虑太阳能电池板的使用寿命和维护费用,因此他的解决方案是错误的。实际解决方案和步骤计算出了正确的首年运营总费用,即450x+100,000美元。因此,学生的解决方案和实际解决方案是否相同:是。学生的成绩:正确。


提示词写的比问题还复杂:
- 土地费用 100x : &#10004;
- 太阳能板 250x : &#10004;
- 维护费用 100000+10x : &#10004;
- 总费用：450x+100000 : &#10008;

---------

### 模型的局限性

&emsp;&emsp;模型并不会完美地记忆所见到的全部信息，因此它并不十分清楚它的知识边界。 这意味着它回答可能会虚构听起来很有道理但实际上不正确的东西。我们将这些捏造的想法称为幻觉。因此，我将向您展示一个例子，在这个例子中模型会产生幻觉。编造一个来自真实牙刷公司的虚构产品名称的描述。


```python
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response, history = get_completion(prompt, [], temperature=0.95)
print(response)
```

    TheAeroGlide UltraSlim Smart Toothbrush by Boie is a professional-quality Toothbrush designed to help improve your smile's hygiene and appearance. It features a sleek and slim design, with a深入-clean philosophy that allows for thorough cleaning of your teeth. The brush is made of high-quality materials and is equipped with advanced technology, including a four-pronged brush head that can effectively clean your teeth in a single pass, and a powerful cleaning system that minimizes the risk of developing gum disease and other health problems. The brush also has a built-in LED light system that illuminates areas of your teeth that need extra attention, allowing you to see where you're making mistakes. Overall, theAeroGlide UltraSlim Smart Toothbrush is a stylish and effective tool for improving your hygiene and smile.



```python
prompt = f"""
告诉我 Boie 公司生产的 AeroGlide UltraSlim Smart Toothbrush 的相关信息
"""
response, history = get_completion(prompt, [], temperature=0.95)
print(response)
```

    Boie 公司生产的AeroGlide UltraSlim Smart Toothbrush是一款由Boie公司推出的高端智能toothbrush。以下是一些相关的信息:
    
    1. 设计:AeroGlide UltraSlim Smart Toothbrush采用先进的设计,具有出色的性能和耐用性。它的长度只有3.6厘米,直径是1.5厘米,可以轻松地穿过牙缝和脸颊,同时不会刮伤皮肤。
    
    2. 功能:AeroGlide UltraSlim Smart Toothbrush还配备了多种智能功能,例如自动清洁模式和智能旋转控制。它可以根据不同的场景和需求自动调整清洁模式,确保每次使用都能够彻底清洁牙齿。
    
    3. 材料:AeroGlide UltraSlim Smart Toothbrush采用高品质材料制作,包括不锈钢材料和柔软的硅胶材料。这种材料能够提供卓越的性能和耐用性,同时不会损坏牙齿和牙龈。
    
    4. 价格:AeroGlide UltraSlim Smart Toothbrush的价格相对较高,是一款高端智能toothbrush,适合那些追求高品质和高性能的口腔护理用户。
    
    5. 品牌保障:Boie公司是一家拥有悠久历史和高品质产品的品牌,其产品在全球范围内都备受好评。因此,如果正在寻找一款高品质的智能toothbrush,Boie公司的AeroGlide UltraSlim Smart Toothbrush是一个不错的选择。

