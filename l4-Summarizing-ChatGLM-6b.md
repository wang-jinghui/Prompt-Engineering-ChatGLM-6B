# 文本总结

&emsp;&emsp;当今世界有如此多的文本信息，阅读需要花费大量的时间，但是如果可以对文本的内容进行压缩，去掉冗余的内容保留你感兴趣的重点信息，这样就能节省不少时间，同时还提高了我们的阅读效率。大语言模型可以帮助我们自动的实现文本内容的提取。 

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


    CPU times: user 2 µs, sys: 1 µs, total: 3 µs
    Wall time: 5.25 µs


    Explicitly passing a `revision` is encouraged when loading a configuration with custom code to ensure no malicious code has been contributed in a newer revision.
    Explicitly passing a `revision` is encouraged when loading a model with custom code to ensure no malicious code has been contributed in a newer revision.



    Loading checkpoint shards:   0%|          | 0/8 [00:00<?, ?it/s]



```python
def get_completion(prompt, history, temperature=0.95):
    
    response, history = model.chat(tokenizer=tokenizer, query=prompt, history=history,
                                   temperature=temperature)
    return response, history
```

### 任务1：文本内容总结

&emsp;&emsp;任务是对产品评论内容进行总结，评论内容："我得到了这个熊猫毛绒玩具作为女儿生日礼物，她非常喜欢并且带它到处走等等"。假设你运营着一个电子商务网站并且有大量的评论，那么一个可以总结冗长评论的工具可以让你很快地浏览更多的评论，以更好地了解所有客户的想法。


```python
prod_review_zh = """
这个熊猫公仔是我给女儿的生日礼物，她很喜欢，去哪都带着。\
公仔很软，超级可爱，面部表情也很和善。但是相比于价钱来说，\
它有点小，我感觉在别的地方用同样的价钱能买到更大的。\
快递比预期提前了一天到货，所以在送给女儿之前，我自己玩了会。
"""
```


```python
prompt = f"""
你的任务是对一个产品评论内容进行一个简短的概括。

请对三个反引号之间的评论文本进行概括，最多30个词。

评论: ```{prod_review_zh}```
"""

response, history = get_completion(prompt, [])
print(response)
```

    概括:这是一款送给女儿的生日礼物,她非常喜欢。虽然公仔有点小,但其他特征都很符合期望。快递速度比预期快,自己先玩了一下。


### 任务2：有针对性的内容总结

修改prompt侧重对用户评论中物流信息的总结：


```python
prompt = f"""
你的任务是对一个产品评论内容进行一个简短的概括。

请对三个反引号之间的评论文本进行概括，最多30个词，聚焦产品的运输。

评论: ```{prod_review_zh}```
"""
response, history = get_completion(prompt, history)
print(response)
```

    概括: 好评,熊猫公仔可爱,小但实用,快递提前到达。



```python
prompt = f"""
你的任务是对一个产品评论内容进行一个简短的总结。

请对三个反引号之间的评论文本进行总结，最多30个词，聚焦产品的价格。

评论: ```{prod_review_zh}```
"""
response, history = get_completion(prompt, history)
print(response)
```

    总结: 
    
    这个熊猫公仔很适合作为女儿生日礼物,软、可爱、和善。虽然公仔有点小,但也可以在其他地方找到更大更便宜的替代品。快递速度比预期提前了一天,但在送给女儿之前,自己玩了会。


### 任务3：尝试提取信息

&emsp;&emsp;在上面的例子中，通过修改提示文本让文本总结侧重于某一特定方面，但是结果中仍会保留一些其他信息。如果我们只需要提取某一特定方面的信息，并过滤掉其他信息，可以通过修改提示要求LLM进行“文本提取(Extract)”而非“文本概括(Summarize)”。


```python
prompt = f"""
你的任务是对电子商务网站上的一个产品评论提取相关信息。

请从以下三个反引号之间的评论文本中只提取快递相关的信息，最多30个词汇。

评论: ```{prod_review_zh}```
"""
response, history = get_completion(prompt, [])
print(response)
```

    快递:
    - 快递比预期提前了一天到货
    - 熊猫公仔快递速度较快,服务态度好


### 任务4：对多项信息进行总结

&emsp;&emsp;下面基于for循环调用模型对多个评论进行总结。当然，在实际生产中，对于上百万甚至上千万的评论文本，使用for循环也是不现实的，可能需要考虑整合评论、分布式等方法提升运算效率。


```python
review_1 = f"""
我的卧室需要一盏漂亮的灯，这盏灯有额外的储物空间，而且价格不太高。\
很快2天内到达。灯线在运输过程中断了，公司很高兴地送来了一根新的。\
几天之内也来了。很容易放在一起。\
然后我缺少了一部分，所以我联系了他们的支持，他们很快就帮我找到了丢失的部分！\
在我看来，这是一家关心客户和产品的伟大公司。
"""
```


```python
review_2 = f"""
我的牙科保健员推荐了电动牙刷，这就是我买这个的原因。到目前为止，电池寿命似乎令人印象深刻。在初次充电并在第一周保持充电器插入状态以调节电池后，我拔下充电器并在过去的3周内每天使用它刷牙两次，每次充电都是一样的。但是牙刷头太小了。我见过比这个大的婴儿牙刷。我希望头部更大，刷毛长度不同，以便更好地进入牙齿之间，因为这个没有。总的来说，如果你能在50美元左右买到这个，那还是很划算的。制造商的替换头非常昂贵，但您可以获得价格更合理的通用头。这把牙刷让我觉得我每天都去看牙医了。我的牙齿感觉闪闪发光！
"""
```


```python
review_3 = f"""
他们在11月份仍然以49美元左右的价格对17件系统进行季节性销售，大约减半，但由于某种原因（称之为价格欺诈），在12月的第二周左右，价格全部上涨至大约任何地方 同一系统在70-89美元之间。11件套系统的价格也比之前的29美元上涨了10美元左右。所以它看起来还不错，但是如果你看一下底座，刀片锁定到位的部分看起来不像几年前的以前版本那么好，但我打算对它非常温和（例如，我 先在搅拌机中粉碎非常硬的东西，如豆子、冰、大米等，然后在搅拌机中将它们粉碎成我想要的份量，然后切换到搅打刀片以获得更细的面粉，并在制作冰沙时先使用横切刀片 ，然后如果我需要它们更细/更少浆状，请使用平刀片）。制作冰沙时的特别提示，切碎并冷冻您计划使用的水果和蔬菜（如果使用菠菜 - 稍微炖软菠菜，然后冷冻直至准备使用 - 如果制作冰糕，请使用中小型食品加工机） 这样你就可以避免在制作冰沙时添加太多冰块。大约一年后，电机发出一种奇怪的声音。 我打电话给客户服务，但保修期已经过期，所以我不得不再买一个。仅供参考：这些类型的产品的整体质量已经过时，因此他们有点依赖品牌知名度和消费者忠诚度来维持销售。大概两天就拿到了
"""
```


```python
reviews = [review_1, review_2, review_3]
```


```python
for i in range(len(reviews)):
    prompt = f"""
     你的任务是对一个产品评论内容进行一个简短的概括。

     请对三个反引号之间的评论文本进行概括，最多20个词。

    评论: ```{reviews[i]}```
    """
    response,history = get_completion(prompt, [])
    print(i, response, "\n")
```

    0 概括:赞扬该产品快速、高质量地满足了需求,关心客户,并提供良好的支持。 
    
    1 评论:电动牙刷不错,电池寿命令人印象深刻,但牙刷头太小,希望头部更大,刷毛长度不同。价格划算,替换头昂贵,但可以购买更合理的通用头。觉得每天都去看牙医了,牙齿感觉闪闪发光! 
    
    2 评论:该评论对一款产品进行了简要介绍和评价。评论指出,该产品的质量过时,依赖品牌知名度和消费者忠诚度来维持销售。此外,评论还指出,该产品的底座刀片锁定不到位,需要温和对待使用。最后,评论提醒注意产品保修期的过期问题,并建议消费者购买其他产品。 



### 英文提示


```python
prod_review = """
Got this panda plush toy for my daughter's birthday, \
who loves it and takes it everywhere. It's soft and \ 
super cute, and its face has a friendly look. It's \ 
a bit small for what I paid though. I think there \ 
might be other options that are bigger for the \ 
same price. It arrived a day earlier than expected, \ 
so I got to play with it myself before I gave it \ 
to her.
"""
```

#### Summarize with a word/sentence/character limit


```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words. 

Review: ```{prod_review}```
"""

response, _ = get_completion(prompt, [])
print(response)
```

    The product review is about a pandas plush toy that was given to a daughter for her birthday. The toy is soft and cute, with a friendly face. It's a bit small for the price, but there might be other options for the same price. The toy arrived a day earlier than expected, so the girl had time to play with it before it was given to her.


#### Summarize with a focus on shipping and delivery


```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
Shipping deparmtment. 

Summarize the review below, delimited by triple 
backticks, in at most 30 words, and focusing on any aspects \
that mention shipping and delivery of the product. 

Review: ```{prod_review}```
"""

response,_= get_completion(prompt,[])
print(response)
```

    Overall, the product was a success. The plush toy is soft and cute, and the face of the panda is friendly. However, it is small for the price that I paid. I think there might be other options that are bigger for the same price. The product arrived early, so I got to play with it before giving it to my daughter. Overall, I would recommend this product to others.


没有突出物流特别快，只提到一句：**The product arrived early**

#### Summarize with a focus on price and value


```python
prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
pricing deparmtment, responsible for determining the \
price of the product.  

Summarize the review below, delimited by triple 
backticks, in at most 30 words, and focusing on any aspects \
that are relevant to the price and perceived value. 

Review: ```{prod_review}```
"""

response,_ = get_completion(prompt,[])
print(response)
```

    This review describes a product from an ecommerce site, where the review concerns the price and perceived value of the product. The review includes the following information:
    
    -   The product is a panda plush toy for a child's birthday
    -   The product is soft and cute, with a friendly face
    -   The product arrived early, allowing the reviewer to play with it before giving it to their child
    -   The product is considered to be small for the price, but the reviewer likes that it allows their child to take it with them.


总结没有突出重点，跟原文还长。

#### Try "extract" instead of "summarize"


```python
prompt = f"""
Your task is to extract relevant information from \ 
a product review from an ecommerce site to give \
feedback to the Shipping department. 

From the review below, delimited by triple quotes \
extract the information relevant to shipping and \ 
delivery. Limit to 30 words. 

Review: ```{prod_review}```
"""

response,_ = get_completion(prompt,[])
print(response)
```

    shipping was fine, but the product itself was a bit small for the price. I would suggest finding a larger size option. The toy arrived on time, which is good.


#### Summarize multiple product reviews


```python
review_1 = prod_review 

# review for a standing lamp
review_2 = """
Needed a nice lamp for my bedroom, and this one \
had additional storage and not too high of a price \
point. Got it fast - arrived in 2 days. The string \
to the lamp broke during the transit and the company \
happily sent over a new one. Came within a few days \
as well. It was easy to put together. Then I had a \
missing part, so I contacted their support and they \
very quickly got me the missing piece! Seems to me \
to be a great company that cares about their customers \
and products. 
"""

# review for an electric toothbrush
review_3 = """
My dental hygienist recommended an electric toothbrush, \
which is why I got this. The battery life seems to be \
pretty impressive so far. After initial charging and \
leaving the charger plugged in for the first week to \
condition the battery, I've unplugged the charger and \
been using it for twice daily brushing for the last \
3 weeks all on the same charge. But the toothbrush head \
is too small. I’ve seen baby toothbrushes bigger than \
this one. I wish the head was bigger with different \
length bristles to get between teeth better because \
this one doesn’t.  Overall if you can get this one \
around the $50 mark, it's a good deal. The manufactuer's \
replacements heads are pretty expensive, but you can \
get generic ones that're more reasonably priced. This \
toothbrush makes me feel like I've been to the dentist \
every day. My teeth feel sparkly clean! 
"""

# review for a blender
review_4 = """
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \ 
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \ 
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \ 
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

reviews = [review_1, review_2, review_3, review_4]
```


```python
for i in range(len(reviews)):
    prompt = f"""
    Your task is to generate a short summary of a product \ 
    review from an ecommerce site. 

    Summarize the review below, delimited by triple \
    backticks in at most 20 words. 

    Review: ```{reviews[i]}```
    """

    response,_ = get_completion(prompt,[])
    print(i, response, "\n")
```

    0 The review is of a ecommerce site\'s review of a panda plush toy for a birthday gift for a daughter. The toy is soft and cute, and the face is friendly. It\'s a bit small for the price, but the reviewer feels there might be other options that are bigger for the same price. The toy arrived early, so the reviewer got to play with it before giving it to their daughter. 
    
    1 This review describes a product review from an ecommerce site. The review reads as follows:
    
        This lamp was a great purchase for me as I needed a nice lamp for my bedroom. It had additional storage and was not too high of a price point. The lamp arrived fast with the string broken, but the company sent over a new one quickly. It was easy to put together and the support team was very helpful. I appreciate the company\'s attention to their customers and products. 
    
    2 This electric toothbrush from an ecommerce site is recommended by a dental hygienist. The battery life is impressive, and it can be used twice daily on the same charge for a week. However, the toothbrush head is too small, and you may need to use different length bristles to get between teeth better. The prices for the manufacturer\'s replacement heads are higher, but you can get generic ones for less. Overall, if you can get this one for around the $50 mark, it\'s a good deal. The feeling of being clean and干燥 after using this brush is great. 
    
    3 This review describes a product from an Ecommerce site, including its price变化 and customer reviews. The summary highlights the problems with the product, such as the Base part where the Blade Locks into place not as good as previous Editions, and the customer service issue with the warranty. The review also mentions the overall quality of the product has gone down in recent years and that the brand is counting on brand recognition and consumer loyalty to maintain sales. The product was able to be found and purchased in about two days. 



### 总结：

- 中文内容总结 ：&#10004;
- 中文侧重总结 ：&#10004;
- 中文信息提取 ：&#10004;
- 英文内容总结 ：&#10008;,经常夹杂着中文字符输出。
