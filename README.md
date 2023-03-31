# langchain-experiments
对于个人使用来说，调接口并维护memory是一件极其奢侈的事情，但是给给公司用搞钱还是很不错的，反正别自己花钱维护对话历史就好了。      

![image](https://user-images.githubusercontent.com/17697154/229182819-3da19c61-62ed-4d80-ae16-82bf3fd95c0b.png)



```python
# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE = """You are an assistant to a human, powered by a large language model trained by OpenAI.
You are designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, you are able to generate human-like text based on the input you receive, allowing you to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
You are constantly learning and improving, and your capabilities are constantly evolving. You are able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. You have access to some personalized information provided by the human in the Context section below. Additionally, you are able to generate your own text based on the input you receive, allowing you to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, you are a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether the human needs help with a specific question or just wants to have a conversation about a particular topic, you are here to assist.
Context:
{entities}
Current conversation:
{history}
Last line:
Human: {input}
You:"""

ENTITY_MEMORY_CONVERSATION_TEMPLATE = PromptTemplate(
    input_variables=["entities", "history", "input"],
    template=_DEFAULT_ENTITY_MEMORY_CONVERSATION_TEMPLATE,
)
```
```txt
你是由OpenAI训练的大型语言模型提供技术支持的人工智能助手。你的设计旨在能够协助完成各种任务，
从简单问题的回答到提供深入的解释和广泛讨论各种主题。    
作为一个语言模型，你能够根据接收到的输入生成类人的文本，使你能够进行自然而流畅的对话，并提供连贯、相关的回应。    
你不断地学习和进化，你的能力也在不断提升。你能够处理和理解大量的文本，并利用这些知识为各种问题提供准确、有益的回答。    
你可以访问由人类在下文中提供的一些个性化信息。此外，你还能根据接收到的输入生成自己的文本，使你能够进行讨论，并对各种主题提供解释和描述。    
总体而言，你是一种强大的工具，可以帮助完成各种任务，并为各种主题提供有价值的见解和信息。无论人类需要回答特定问题，
还是想就某个特定主题进行对话，你都会在这里提供帮助。

上下文:
{entities}
当前对话:
{history}
最后一行：
人类: {input}
你: """
```

```python
_DEFAULT_SUMMARIZER_TEMPLATE = """Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.
New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.
New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE
Current summary:
{summary}
New lines of conversation:
{new_lines}
New summary:"""
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)
```

```txt
逐步总结提供的对话内容，将新的摘要添加到以前的摘要中，返回一个新的摘要。

例子：
当前摘要：
人类问人工智能对人工智能的看法。 人工智能认为人工智能是一种有益力量。
新的对话行：
人类：你为什么认为人工智能是有益的力量？
人工智能：因为人工智能将帮助人类发挥他们的全部潜力。
新的摘要：
人类问人工智能对人工智能的看法。 人工智能认为人工智能是一种有益力量，因为它将帮助人类发挥他们的全部潜力。
例子结束

当前摘要：
{summary}
新的对话行：
{new_lines}
新的摘要：
```

```python
_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the proper nouns from the last line of conversation. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places.
The conversation history is provided just in case of a coreference (e.g. "What do you know about him" where "him" is defined in a previous line) -- ignore items mentioned there that are not in the last line.
Return the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is just issuing a greeting or having a simple conversation).
EXAMPLE
Conversation history:
Person #1: how's it going today?
AI: "It's going great! How about you?"
Person #1: good! busy working on Langchain. lots to do.
AI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"
Last line:
Person #1: i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
Output: Langchain
END OF EXAMPLE
EXAMPLE
Conversation history:
Person #1: how's it going today?
AI: "It's going great! How about you?"
Person #1: good! busy working on Langchain. lots to do.
AI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"
Last line:
Person #1: i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Person #2.
Output: Langchain, Person #2
END OF EXAMPLE
Conversation history (for reference only):
{history}
Last line of conversation (for extraction):
Human: {input}
Output:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)
```

```txt
你是一位 AI 助手，正在阅读 AI 和人类之间对话的记录。   
提取出最后一行对话中的所有专有名词。作为指南，专有名词通常大写。您应该提取所有名称和地点。   
为了防止指代问题（例如，“你对他了解多少”中的“他”在前一行中定义）提供了对话历史记录--忽略其中在最后一行未提到的内容。  
将输出作为单个逗号分隔的列表返回，如果没有要返回的内容（例如，用户只是打招呼或进行简单对话），则返回 NONE。

例子：
对话历史：
人 #1：今天怎么样？
AI：“很好！你呢？”
人 #1：好！忙着开发 Langchain。有很多事要做。
AI：“听起来很辛苦！你做了哪些事情让 Langchain 变得更好？”
最后一行：
人 #1：我正在尝试改进 Langchain 的界面、用户体验、与用户可能需要的各种产品的集成...很多东西。
输出：Langchain
例子结束

例子：
对话历史：
人 #1：今天怎么样？
AI：“很好！你呢？”
人 #1：好！忙着开发 Langchain。有很多事要做。
AI：“听起来很辛苦！你做了哪些事情让 Langchain 变得更好？”
最后一行：
人 #1：我正在尝试改进 Langchain 的界面、用户体验、与用户可能需要的各种产品的集成...很多东西。我正在与人 #2 合作。
输出：Langchain，人 #2
例子结束

对话历史（仅供参考）：
{history}
对话中的最后一行（用于提取）：
人类：{input}
输出：
```

```python
_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE = """You are an AI assistant helping a human keep track of facts about relevant people, places, and concepts in their life. Update the summary of the provided entity in the "Entity" section based on the last line of your conversation with the human. If you are writing the summary for the first time, return a single sentence.
The update should only include facts that are relayed in the last line of conversation about the provided entity, and should only contain facts about the provided entity.
If there is no new information about the provided entity or the information is not worth noting (not an important or relevant fact to remember long-term), return the existing summary unchanged.
Full conversation history (for context):
{history}
Entity to summarize:
{entity}
Existing summary of {entity}:
{summary}
Last line of conversation:
Human: {input}
Updated summary:"""

ENTITY_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["entity", "summary", "history", "input"],
    template=_DEFAULT_ENTITY_SUMMARIZATION_TEMPLATE,
)
```

```txt
你是一位 AI 助手，正在帮助人类追踪他们生活中相关人物、地点和概念的信息。根据您与人类的最后一行对话，
更新“实体”部分提供的实体的摘要。如果您是第一次编写摘要，则返回一个句子。
更新应仅包括有关提供的实体的最后一行对话中传达的事实，并且应仅包含有关提供的实体的事实。
如果提供的实体没有新信息或信息不值得注意（不是长期记忆的重要或相关事实），则返回现有摘要而不变。
完整的对话历史记录（供参考）：
{history}
要总结的实体：
{entity}
{entity} 的现有摘要：
{summary}
对话的最后一行：
人类：{input}
更新的摘要：
```

```python
KG_TRIPLE_DELIMITER = "<|>"
_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the last line of conversation."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property.\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "Person #1: Did you hear aliens landed in Area 51?\n"
    "AI: No, I didn't hear that. What do you know about Area 51?\n"
    "Person #1: It's a secret military base in Nevada.\n"
    "AI: What do you know about Nevada?\n"
    "Last line of conversation:\n"
    "Person #1: It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "Person #1: Hello.\n"
    "AI: Hi! How are you?\n"
    "Person #1: I'm good. How are you?\n"
    "AI: I'm good too.\n"
    "Last line of conversation:\n"
    "Person #1: I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "Person #1: What do you know about Descartes?\n"
    "AI: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.\n"
    "Person #1: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.\n"
    "AI: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.\n"
    "Last line of conversation:\n"
    "Person #1: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "Conversation history (for reference only):\n"
    "{history}"
    "\nLast line of conversation (for extraction):\n"
    "Human: {input}\n\n"
    "Output:"
)

KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=_DEFAULT_KNOWLEDGE_TRIPLE_EXTRACTION_TEMPLATE,
)
```

```txt
你是一位联网智能，正在帮助人类追踪与所有相关人物、事物、概念等相关的知识三元组，并将它们与存储在您的权重内的知识以及知识图中存储的知识整合起来。从对话的最后一行中提取所有的知识三元组。知识三元组是包含主语、谓语和宾语的从句。主语是被描述的实体，谓语是被描述的主语的属性，宾语是属性的值。

例子
对话历史：
人 #1：你听说外星人降落在 51 区了吗？
AI：没有，我没听说过。你知道 51 区的情况吗？
人 #1：它是内华达州的一个秘密军事基地。
AI：你知道内华达州的情况吗？
对话的最后一行：
人 #1：它是美国的一个州。它还是美国黄金产量排名第一的州。

输出：(内华达州, 是一个, 州)<|>(内华达州, 在, 美国)<|>(内华达州, 是黄金的第一生产国)
例子结束

例子
对话历史：
人 #1：你好。
AI：嗨！你好吗？
人 #1：我很好。你呢？
AI：我也很好。
对话的最后一行：
人 #1：我要去商店。

输出：NONE
例子结束

例子
对话历史：
人 #1：你对笛卡尔了解多少？
AI：笛卡尔是一位法国哲学家、数学家和科学家，生活在17世纪。
人 #1：我提到的笛卡尔是一位蒙特利尔的脱口秀演员和室内设计师。
AI：哦，是的，他是一位脱口秀演员和室内设计师。他已经在这个行业工作了30年。他最喜欢的食物是烘豆饼。
对话的最后一行：
人 #1：哦，我知道笛卡尔喜欢开古董摩托车和弹曲琴。

输出：(笛卡尔, 喜欢开, 古董摩托车)<|>(笛卡尔, 弹, 曲琴)
例子结束

对话历史（仅供参考）：
{history}
对话的最后一行（用于提取）：
人类：{input}

输出：
```







