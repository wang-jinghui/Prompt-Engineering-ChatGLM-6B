# <center> 提示工程指南：提示工程迭代
    
&emsp;&emsp;在使用大型语言模型构建应用程序时，我们需要有一个良好的迭代过程来改进提示，逐渐找到能够完成所需任务的提示。最开始您有一个想法，想要完成某个具体的任务，尝试编写第一个提示，然后运行它并查看结果。如果第一次效果不好，则迭代的过程中可以找出原因，例如：指示不够清晰或算法没有足够的时间进行思考，从而改进提示，如此循环多次，直到开发应用程序所需的提示得以完成。这跟机器学习开发的流程很像。通常是先有一个想法，然后去实现它，得到一个实验结果。然后观察结果，进行误差分析，找出问题，然后更改实现并重新实验等等，反复迭代，最终得到一个有效的模型。

### 一：环境设置

加载开源的**chatGLM**模型,使用ChatGLM-6b的INT8版本。


```python
import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModel
```


```python
%time
tokenizer = AutoTokenizer.from_pretrained("./chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm-6b", trust_remote_code=True).half().quantize(8).to('cuda')
```

    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.


    CPU times: user 5 µs, sys: 0 ns, total: 5 µs
    Wall time: 10 µs


    Explicitly passing a `revision` is encouraged when loading a configuration with custom code to ensure no malicious code has been contributed in a newer revision.
    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.



    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]


显存占用情况：
```
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
| 29%   29C    P8     1W / 250W |   7086MiB / 11019MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```


```python
def get_completion(prompt, history, temperature=0.95):
    
    response, history = model.chat(tokenizer=tokenizer, query=prompt, history=history,
                                   temperature=temperature)
    return response, history
```

### 二 ：任务描述

根据一个产品的信息清单，帮助营销团队撰写产品描述等。


```python
# 示例：产品说明书
fact_sheet_chair = """
概述

    美丽的中世纪风格办公家具系列的一部分，包括文件柜、办公桌、书柜、会议桌等。
    多种外壳颜色和底座涂层可选。
    可选塑料前后靠背装饰（SWC-100）或10种面料和6种皮革的全面装饰（SWC-110）。
    底座涂层选项为：不锈钢、哑光黑色、光泽白色或铬。
    椅子可带或不带扶手。
    适用于家庭或商业场所。
    符合合同使用资格。

结构

    五个轮子的塑料涂层铝底座。
    气动椅子调节，方便升降。

尺寸

    宽度53厘米|20.87英寸
    深度51厘米|20.08英寸
    高度80厘米|31.50英寸
    座椅高度44厘米|17.32英寸
    座椅深度41厘米|16.14英寸

选项

    软地板或硬地板滚轮选项。
    两种座椅泡沫密度可选：中等（1.8磅/立方英尺）或高（2.8磅/立方英尺）。
    无扶手或8个位置PU扶手。

材料
外壳底座滑动件

    改性尼龙PA6/PA66涂层的铸铝。
    外壳厚度：10毫米。
    座椅
    HD36泡沫

原产国
    意大利
"""
```


```python
# 提示
prompt = f"""
你的任务是帮助营销团队基于技术说明书创建一个产品的营销描述。

根据```标记的技术说明书中提供的信息，编写一个简短的产品描述。

技术说明: ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    产品描述:
    
    我们的中世纪风格办公家具系列采用意大利制造,包括文件柜、办公桌、书柜、会议桌等。该系列采用美丽优雅的设计,具有多种颜色和底座涂层选项,可选塑料前后靠背装饰(SWC-100)或10种面料和6种皮革的全面装饰(SWC-110)。该系列椅子可带或不带扶手,适用于家庭或商业场所,并符合合同使用资格。
    
    该系列采用五个轮子的塑料涂层铝底座,气动椅子调节,方便升降。此外,该系列还具有多种尺寸和选项,包括软地板或硬地板滚轮选项、两种座椅泡沫密度可选、无扶手或8个位置PU扶手等。
    
    该系列外壳和底座滑动件采用改性尼龙PA6/PA66涂层的铸铝,座椅采用HD36泡沫材料制造,具有优秀的舒适感和支撑力。
    
    我们提供多种颜色和底座涂层选项,可根据个人需求选择。适用于家庭或商业场所,无论是办公室、卧室还是客厅,都能为您提供完美的家具体验。
    
    原产国:意大利。


#### 问题1：生成文本过长

修改提示限制文本生成的长度：


```python
# 提示
prompt = f"""
你的任务是帮助营销团队基于技术说明书创建一个产品的营销描述。

根据```标记的技术说明书中提供的信息，编写一个简短的产品描述。

描述最多使用50个词。

技术说明: ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, history)
print(response)
```

    中世纪风格办公家具,包括文件柜、办公桌、书柜、会议桌等。有多种外壳颜色和底座涂层可选,可选塑料前后靠背装饰(SWC-100)或10种面料和6种皮革的全面装饰(SWC-110)。底座涂层选项为:不锈钢、哑光黑色、光泽白色或铬。椅子可带或不带扶手,适用于家庭或商业场所。符合合同使用资格。采用改性尼龙PA6/PA66涂层的铸铝外壳和座椅HD36泡沫材料。有多种滚轮选项和无扶手或PU扶手选择。原产国为意大利。



```python
len(response)
```




    202



#### 问题2：关注产品的细节

- 接下来修改这个提示，使其更加精确地描述产品细节。 


```python
# 优化后的 Prompt 
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的营销描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

该描述面向家具零售商，因此应具有技术性质，并侧重于产品的材料构造。

描述最多使用50个词。

技术规格： ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, history)
print(response)
```

    产品描述:
    
    这款办公家具系列是美丽中世纪风格的杰出代表,包括文件柜、办公桌、书柜、会议桌等。该系列采用多种外壳颜色和底座涂层可选,包括不锈钢、哑光黑色、光泽白色和铬。椅子可带或不带扶手,适用于家庭或商业场所,符合合同使用资格。该系列采用五个轮子的塑料涂层铝底座和气动椅子调节,方便升降。尺寸包括宽度53厘米|20.87英寸、深度51厘米|20.08英寸、高度80厘米|31.50英寸和座椅高度44厘米|17.32英寸、座椅深度41厘米|16.14英寸。此外,还有软地板或硬地板滚轮选项和两种座椅泡沫密度可选,无扶手或8个位置PU扶手。该系列原产于意大利。


#### 问题3：表格形式的产品描述

&emsp;&emsp;进一步修改提示，让模型输出一个包含产品尺寸的表格，然后将所有内容格式化为 HTML。在实践中，只有在多次迭代后才能最终得到这样的提示。我不认为有人会在第一次尝试时就写出这这样复杂精确的提示。


```python
# 要求它抽取信息并组织成表格，并指定表格的列、表名和格式
prompt = f"""
您的任务是帮助营销团队基于技术说明书创建一个产品的零售网站描述。

根据```标记的技术说明书中提供的信息，编写一个产品描述。

该描述面向家具零售商，因此应具有技术性质，并侧重于产品的材料构造。

在描述末尾，包括技术规格中每个7个字符的产品ID。

在描述之后，包括一个表格，提供产品的尺寸。表格应该有两列。第一列包括尺寸的名称。第二列只包括英寸的测量值。

给表格命名为“产品尺寸”。

将描述的内容格式化为可用于网站的HTML格式。将描述放在<div>元素中。

技术规格：```{fact_sheet_chair}```
"""

response, history = get_completion(prompt, [])
print(response)
```

    产品描述:
    
    我们的产品是中世纪风格办公家具系列的一部分,包括文件柜、办公桌、书柜、会议桌等。我们提供多种外壳颜色和底座涂层可选,以及塑料前后靠背装饰(SWC-100)或10种面料和6种皮革的全面装饰(SWC-110)。底座涂层选项包括不锈钢、哑光黑色、光泽白色或铬。椅子可带或不带扶手,适用于家庭或商业场所,符合合同使用资格。
    
    我们的产品采用五个轮子的塑料涂层铝底座,气动椅子调节,方便升降。外壳和座椅材料采用改性尼龙PA6/PA66涂层的铸铝,外壳厚度为10毫米,座椅采用HD36泡沫。原产国为意大利。
    
    产品尺寸:
    
    宽度53厘米|20.87英寸
    深度51厘米|20.08英寸
    高度80厘米|31.50英寸
    座椅高度44厘米|17.32英寸
    座椅深度41厘米|16.14英寸
    
    表格:
    
    | 产品ID | 宽度 | 深度 | 高度 | 座椅高度 | 座椅深度 |
    |------|------|----------|----------|----------|
    | SWC-100 | 20.87 | 20.08 | 31.50 | 44 | 41 |
    | SWC-110 | 20.87 | 20.08 | 31.50 | 41 | 41 |
    
    注:宽度是指产品宽度测量值,深度是指产品深度测量值,高度是指产品高度测量值,座椅高度是指座椅高度测量值,座椅深度是指座椅深度测量值。


| 产品ID | 宽度 | 深度 | 高度 | 座椅高度 | 座椅深度 |
|------|------|----------|----------|----------|---|
| SWC-100 | 20.87 | 20.08 | 31.50 | 44 | 41 |
| SWC-110 | 20.87 | 20.08 | 31.50 | 41 | 41 |

- HTML格式：&#10008;
- 信息Table: &#10004;

### 英文版提示


```python
fact_sheet_chair = """
OVERVIEW
- Part of a beautiful family of mid-century inspired office furniture, 
including filing cabinets, desks, bookcases, meeting tables, and more.
- Several options of shell color and base finishes.
- Available with plastic back and front upholstery (SWC-100) 
or full upholstery (SWC-110) in 10 fabric and 6 leather options.
- Base finish options are: stainless steel, matte black, 
gloss white, or chrome.
- Chair is available with or without armrests.
- Suitable for home or business settings.
- Qualified for contract use.

CONSTRUCTION
- 5-wheel plastic coated aluminum base.
- Pneumatic chair adjust for easy raise/lower action.

DIMENSIONS
- WIDTH 53 CM | 20.87”
- DEPTH 51 CM | 20.08”
- HEIGHT 80 CM | 31.50”
- SEAT HEIGHT 44 CM | 17.32”
- SEAT DEPTH 41 CM | 16.14”

OPTIONS
- Soft or hard-floor caster options.
- Two choices of seat foam densities: 
 medium (1.8 lb/ft3) or high (2.8 lb/ft3)
- Armless or 8 position PU armrests 

MATERIALS
SHELL BASE GLIDER
- Cast Aluminum with modified nylon PA6/PA66 coating.
- Shell thickness: 10 mm.
SEAT
- HD36 foam

COUNTRY OF ORIGIN
- Italy
"""
```

- 输出产品描述


```python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Technical specifications: ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    This mid-century inspired office furniture product is part of a beautiful family of office products, including file cabinets, desks, bookcases, and more. It is available in several shell color and base finish options, and is available with or without armrests. The chair is made with a 5-wheel plastic coated aluminum base and is equipped with apneumatic adjustability. The chair has aWIDTH of 53CM | 20.87”, aDepth of 51CM | 20.08”, aHEIGHT of 80CM | 31.50”, aSEATHEIGHT of 44CM | 17.32”, aSEATDEPTH of 41CM | 16.14”, and an options of soft or hard-floor caster options. The chair is made in Italy and is qualified for contract use. The base finish options are: stainless steel, matte black, gloss white, or chrome, and the chair is available with or without armrests. The chair is suitable for home or business settings and is available in several options to suit your personal preferences.


- 限制输出描述文本的长度


```python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    This chair is part of a beautiful family of office furniture designed in the mid-century era and available in several options of shell color and base finish. It is available with plastic back and front upholstery (SWC-100) or full upholstery (SWC-110) in 10 fabric and 6 leather options. The base finish options are: stainless steel, matte black, gloss white, or chrome. The chair is available with or without armrests and is suitable for home or business settings. It is built with a 5-wheel plastic coated aluminum base and is equipped with apneumatic adjustability. The chair has a width of 53CM, a depth of 51CM, a height of 80CM, and a seat height of 44CM and 41CM, respectively. It has a soft or hard-floor caster option and two choices of seat foam density: medium (1.8 lb/ft3) or high (2.8lb/ft3). The chair is made in Italy and is qualified for contract use.


- 关注产品细节


```python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, [])
print(response) 
```

    This chair, part of a beautiful family of office furniture, is a mid-century inspired design with several options of shell color and base finish. It is available with plastic back and front upholstery (SWC-100) or full upholstery (SWC-110) in 10 fabric and 6 leather options. The base finish options are: stainless steel, matte black, gloss white, or chrome. The chair is available with or without armrests and is suitable for home or business settings. It is constructed with a 5-wheel plastic coated aluminum base and can be adjusted with apneumatic action. The dimensions of the chair are:WIDTH 53CM | 20.87”,Depth 51CM | 20.08”,HEIGHT 80CM | 31.50”,SEAT HEIGHT 44CM | 17.32”,SEAT DEPTH 41CM | 16.14”. The materials used in the construction are cast Aluminum with modified nylon PA6/PA66 coating and HD36 foam. The country of origin is Italy.



```python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

At the end of the description, include every 7-character 
Product ID in the technical specification.

Use at most 50 words.

Technical specifications: ```{fact_sheet_chair}```
"""
response, history = get_completion(prompt, [])
print(response) 
```

    `Product ID:` SWC-100


- 格式化输出产品描述：HTML、Table


```python
prompt = f"""
Your task is to help a marketing team create a 
description for a retail website of a product based 
on a technical fact sheet.

Write a product description based on the information 
provided in the technical specifications delimited by 
triple backticks.

The description is intended for furniture retailers, 
so should be technical in nature and focus on the 
materials the product is constructed from.

At the end of the description, include every 7-character 
Product ID in the technical specification.

After the description, include a table that gives the 
product's dimensions. The table should have two columns.
In the first column include the name of the dimension. 
In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website. 
Place the description in a <div> element.

Technical specifications: ```{fact_sheet_chair}```
"""

response, history = get_completion(prompt, [])
print(response) 
```

    ```
    Product Dimensions
    
    - WIDTH 53CM | 20.87”
    - DEPTH 51CM | 20.08”
    - HEIGHT 80CM | 31.50”
    - SEAT HEIGHT 44CM | 17.32”
    - SEAT DEPTH 41CM | 16.14”
    
     options:
    
    - Soft or hard-floor caster options
    - Two choices of seat foam densities:
      - Medium (1.8lb/ft3)
      - High (2.8lb/ft3)
    - Armless or 8 position PU armrests
    
    Material:Cast Aluminum with modified nylon PA6/PA66 coating, HD36 foam
    Country of Origin: Italy
    ```



```python
response
```




    '```\nProduct Dimensions\n\n- WIDTH 53CM | 20.87”\n- DEPTH 51CM | 20.08”\n- HEIGHT 80CM | 31.50”\n- SEAT HEIGHT 44CM | 17.32”\n- SEAT DEPTH 41CM | 16.14”\n\n options:\n\n- Soft or hard-floor caster options\n- Two choices of seat foam densities:\n  - Medium (1.8lb/ft3)\n  - High (2.8lb/ft3)\n- Armless or 8 position PU armrests\n\nMaterial:Cast Aluminum with modified nylon PA6/PA66 coating, HD36 foam\nCountry of Origin: Italy\n```'



### 总结：
- 模型对于中/英文指令中的字符限制，均不敏感。
- chatGML不支持内容的HTML格式化输出。
