# 内容转换

&emsp;&emsp;大型语言模型非常擅长将其输入转换为不同的格式，例如将一种语言中的文本输入并将其转换或翻译成另一种语言，或帮助拼写和语法矫正，或者甚至转换格式，例如输入HTML并输出JSON。因此，我以前用一堆正则表达式写的一堆应用程序现在肯定可以更简单地实现，只需要使用一个大型语言模型和几个提示即可。大型语言模型可以使这些转换过程更加简单和高效，为人们提供更好的应用体验。

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

### 任务1：文本翻译


```python
prompt = f"""
将以下中文翻译成英语: \ 
```您好，我想订购一台电脑。```
"""
response = get_completion(prompt)
print(response)
```

    Hello, I'd like to place an order for a computer.



```python
prompt = f"""
将以下中文翻译成英语: \ 
```您好，我想订购一台搅拌机。```
"""
response = get_completion(prompt)
print(response)
```

    Hello, I'd like to place an order for a搅拌机.



```python
prompt = f"""
将以下中文翻译成英语: \ 
```您好，我想订购一个篮球。```
"""
response = get_completion(prompt)
print(response)
```

    Hello, I'd like to order a basketball.


### 任务2：语种识别


```python
prompt = f"""
请告诉我以下文本是什么语种: 
```Combien coûte le lampadaire?```
"""
response = get_completion(prompt)
print(response)
```

    这个文本不是任何已知的自然语言,因为它包含了一个问句和一个整句话,没有上下文或标记来确定其语言类型。根据文本的内容,它可能是一个机器翻译的结果,或者是某种拼写错误或语法错误。



```python
prompt = f"""
请告诉我以下文本是什么语种: 
```Long time no see```
"""
response = get_completion(prompt)
print(response)
```

    这段文本是中国语(汉语)的感叹句。


### 任务3：语气转换


```python
prompt = f"""
请将以下文本翻译成中文，分别展示成正式与非正式语气: 
```Would you like to order a pillow?```
"""
response = get_completion(prompt)
print(response)
```

    ``` Would you like to order a pillow?```
    
    正式语气: 请问您想要订购一张枕头吗?
    
    非正式语气: 嘿,您想预订一张枕头吗?


### 任务4：风格转换

提示是将以下俚语翻译成商务书信。 


```python
prompt = f"""
将以下文本翻译成商务信函的格式: 
```小李，我是老黄，上回你说咱部门要采购的显卡是多少张来着? 我想在确认一下。```
"""
response = get_completion(prompt)
print(response)
```

    ```
    Dear Li Xiaoling,
    
    I am Li Hua, the company's general manager. In our previous conversation, we discussed the need for the purchase of graphics cards for our department. Could you please confirm the number of graphics cards that we are currently seeking?
    
    Thank you for your assistance.
    
    Best regards,
    
    Li Hua



```python
prompt = f"""
将以下文本修改成商务信函的格式: 
```小李，我是老黄，上回你说咱部门要采购的显卡是多少张来着?，我想在确认一下。```
"""
response = get_completion(prompt)
print(response)
```

    尊敬的小李:
    
    我是老黄,上回咱们部门要采购的显卡数量是多少张来着?我想再次确认一下。
    
    根据我们的计划,我们部门需要采购一定数量的显卡用于项目开发。我们希望尽快完成这项采购工作,因此希望我们能够提前沟通并确定具体的数量。
    
    如果有任何疑问或需要更多信息,请随时与我联系。谢谢!
    
    此致
    
    敬礼
    
    老黄


### 任务5：格式转换

例如JSON到HTML，XML，Markdown等格式，在下面提示中，我们将描述输入和输出格式。 


```python
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}
```


```python
prompt = f"""
将以下Python字典:{data_json},从JSON转换为HTML表格，保留表格标题和列名：

"""
response = get_completion(prompt)
print(response)
```

    以下是将Python字典转换为HTML表格的代码:
    
    ```python
    import json
    
    # 读取JSON数据
    data = {'resturant employees': [{'name': 'Shyam', 'email': 'shyamjaiswal@gmail.com'}, {'name': 'Bob', 'email': 'bob32@gmail.com'}, {'name': 'Jai', 'email': 'jai87@gmail.com'}]}
    
    # 将Python字典转换为JSON对象
    json_data = json.loads(data)
    
    # 创建HTML表格
    table = HTMLTableElement()
    
    # 添加表格标题和列名
    table.title = 'Resturant employees'
    table.columns = [HTMLColumnElement(name='name', width=15), HTMLColumnElement(name='email', width=15)]
    
    # 添加表格数据
    for row in json_data['resturant employees']:
        for cell in row:
            table.columns[-1].cell.innerHTML = cell.name
            table.columns[-1].cell.innerHTML = cell.email
    
    # 渲染HTML表格
    table.draw()
    ```
    
    运行这段代码,将得到以下HTML表格:
    
    ```
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Resturant employees</title>
    </head>
    <body>
        <table>
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Email</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Shyam</td>
                    <td>shyamjaiswal@gmail.com</td>
                </tr>
                <tr>
                    <td>Bob</td>
                    <td>bob32@gmail.com</td>
                </tr>
                <tr>
                    <td>Jai</td>
                    <td>jai87@gmail.com</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    ```
    
    在这个表格中,每个元素都是一个HTML表格单元格,包括标题、列名和数据。你可以根据需要修改列名和数据,以适应你的具体需求。



```python
from IPython.display import HTML,display
```


```python
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Resturant employees</title>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Shyam</td>
                <td>shyamjaiswal@gmail.com</td>
            </tr>
            <tr>
                <td>Bob</td>
                <td>bob32@gmail.com</td>
            </tr>
            <tr>
                <td>Jai</td>
                <td>jai87@gmail.com</td>
            </tr>
        </tbody>
    </table>
</body>
</html>
"""
```


```python
display(HTML(html))
```



<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Resturant employees</title>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Shyam</td>
                <td>shyamjaiswal@gmail.com</td>
            </tr>
            <tr>
                <td>Bob</td>
                <td>bob32@gmail.com</td>
            </tr>
            <tr>
                <td>Jai</td>
                <td>jai87@gmail.com</td>
            </tr>
        </tbody>
    </table>
</body>
</html>



### 任务6：拼写检查

我们将复制一些具有语法或拼写错误的句子列表，然后逐一循环这些句子。请模型校对和纠正这些错误。然后我们将使用一些分隔符。我们将得到响应并像往常一样进行打印。因此，该模型能够纠正所有这些语法错误。


```python
text = [ 
  "The girl with the black and white puppies have a ball.",  # The girl has a ball.
  "Yolanda has her notebook.", # ok
  "Its going to be a long day. Does the car need it’s oil changed?",  # Homonyms
  "Their goes my freedom. There going to bring they’re suitcases.",  # Homonyms
  "Your going to need you’re notebook.",  # Homonyms
  "That medicine effects my ability to sleep. Have you heard of the butterfly affect?", # Homonyms
  "This phrase is to cherck chatGPT for spelling abilitty"  # spelling
]
```


```python
for i in range(len(text)):
    prompt = f"""请校对并更正以下文本，注意纠正文本保持原始语种，无需输出原始文本。
    如果您没有发现任何错误，请说“未发现错误”。
    ```{text[i]}```"""
    response = get_completion(prompt)
    print(i, response)
```

    0 ```The girl with black and white puppies have a ball.未发现错误。```
    1 ```Yolanda has her notebook.```
    
    校对后:
    ```Yolanda has her notebook.```
    
    更正说明:
    - 原句中的“ her”应该改为“ her”以符合英语语法。
    - 原句中的“ notebook”应该改为“ notebook”以符合英语语法。
    2 ``` its going to be a long day. Does the car need it’s oil changed? ```
    
    更正说明:
    1. 将“its”改为“it”;
    2. 将“day”改为“day”,因为“its”用于形容物品,而“day”是名词;
    3. 将“ Does the car need it’s oil changed?”改为“Do you need to change the car’s oil?”,因为“do”用于疑问句中,而“do you need to change the car’s oil?”是陈述句;
    4. 在“ Does the car need it’s oil changed?”后面加上了“?”。
    3 ```
    There goes my freedom. They're going to bring their suitcases.
    ```
    
    更正说明:
    - 原句中的"there"应该改为"there are","there goes"是错误的表达方式;
    - "Their"应该改为"they're","there going to bring"是错误的语法结构;
    - 原句中的"they're suitcases"应该改为"they're cases","there going to bring"也是错误的表达方式。
    4 您的文本已经更正,请参考以下示例:
    
    如果您没有发现任何错误,请说“未发现错误”。
    ```
    Your going to need your're notebook.
    ```
    5 ```That medicine effects my ability to sleep. Have you heard of the butterfly affect?```
    
    更正说明:
    - 原句中的“my ability to sleep”应该改为“my ability to sleep”
    - 原句中的“ Have you heard of the butterfly affect?”应该改为“ Have you heard of the butterfly affect?”
    - 原句中的“that medicine”应该改为“the medicine”
    - 原句中的“ effects”应该改为“影响了”
    - 原句中的“my ability to sleep”和“ Have you heard of the butterfly affect?”之间应该插入一个空格
    - 原句中的“the butterfly affect”应该改为“the蝴蝶效应”(因为“ butterfly”是单词“蝴蝶”的缩写)
    6 ```This phrase is to check the spelling ability of chatGPT.```
    
    注意:在英语中,短语“check the spelling ability of”通常需要拼写正确。因此,我将修正了该短语。


### 总结：
- 中英翻译：&#10008;,中英互译遇到没有的词会输出中文词汇。
- 语种识别：&#10008;
- 语气转换：&#10004;
- 风格转换：&#10004;
- 格式转换：&#10004;
- 拼写检查：&#10008;,chatGLM-6B确实不擅长英语。
