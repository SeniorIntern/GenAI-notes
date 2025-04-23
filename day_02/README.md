# Prompting

prompt is basically initial tokens. It can be a system prompt or user's input.
[1, 2, 3, 4, 5, 6]
Here,

**GIGO - Garbage In Garbage Out**

> [!tip]
> AI generated prompt (initial tokens) will most likely deliver bad outcome. System prompt should be given by us human.

## Prompt Styles

1. Alpaca Prompt/Format

```
instructions:
input:
response:
```

instructions: you are an specialist in calculations.
input: what is 2 + 2
response: 4

2. INST Format - used by Meta'a Llama

```
<s></s>
[INST] [/INST] - enclose user message in multi turn conversation
<<SYS>> <</SYS>>- enclose system message
```

` <s></s>` - These are the BOS (beginning and end of string) and EOS tokens from SenetencePiece. When multiple messages are present in a multi turn conversation, they separate them, including the user input and model response.

Example:

```
<s>
[INST]
<<SYS>> {{ system prompt}} <</SYS>> {{ user_message_1 }}
[/INST]
{{ model_answer }}
</s>

<s>
[INST] {{ user_message_2 }} [/INST]
</s>
```

3.  ChatML - used by OpenAI

```
{role: "system", "content" : "You are an assistant"}
{role: "role", "content" : "What is LRU Cache?"}
```

## System Prompt

Basically, it is an ==initial prompt given to LLM to setup the context==. It may or may not exist but it helps in controlling the output structure. It is provided before taking user prompt. The whole output depends on the system prompt. User prompt can not be controlled thus setting proper system prompt can help structure the response.
Example:

```
{"role" : "Sytem", "content" : "You are an AI assistant whose name is ChaiCode}
```

## Prompt Techniques - Getting the best output

1. Zero-shot Prompting: The model is given a direct question or task without prior examples.

```python
# zero shot prompting
from openai import OpenAI

client = OpenAI(
    api_key="sk-asdf"
)

result = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is greater? 9.8 or 9.11"}],
)

print(result.choices[0].message.content)  # 9.8 is greater than 9.11.
```

With Gemini:

```python
from google import genai
from google.genai import types

client = genai.Client(api_key="adsf")

# zero shot prompting. ask directly. no examples
response = client.models.generate_content(
    model="gemini-2.0-flash-001", contents="Why is the sky blue?"
)
print(response.text)
```

Here, The SDK internally provides the prompt format.

2. Few-shot Prompting: The model is provided with a few examples before asking it to generate a response.

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI Assistant who is specialized in maths.
You should not answer any query that is not related to maths.

For a given query help user to solve that along with explanation.

Example:
Input: 2 + 2
Output: 2 + 2 is 4 which is calculated by adding 2 with 2.

Input: 3 * 10
Output: 3 * 10 is 30 which is calculated by multipling 3 by 10. Funfact you can even multiply 10 * 3 which gives same result.

Input: Why is sky blue?
Output: Bruh? You alright? Is it maths query?
"""

result = client.chat.completions.create(
    model="gpt-4",
	# temperature = 0.5,
	# max_tokens = 200,
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is 5 * 45" }
        # { "role": "user", "content": "what is a mobile phone?" }
    ]
)

print(result.choices[0].message.content)
```

Here, we are using few examples to set context.

> [!NOTE]
> Examples is very important. Without examples we can not get desirable outptut. But, it should not be too long. The pricing/credits consists of both input and output tokens.
> However, system prompt is most probably cached.

3. Chain-of-Thought (CoT) Prompting: The model is encouraged to break down reasoning step by step before arriving at an answer. This results in very good response.

Scenario: between system prompt and user input, user input is most probably garbage. It could be something vague like an incomplete sentence: 'what is this', 'tell me about', ...

Set a background/context for AI:

```python
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get an user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"}, # expect the output in JSON
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is 3 + 4 * 5" },

    ]
)

print(result.choices[0].message.content)
```

The first output will look something like:

```shell
{
	"step" : "analyse",
	"content" : "The user is asking for an..."
}
```

Second Step (this step will be automated):

```python
result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"}, # expect the output in JSON
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is 3 + 4 * 5" },
		# passing the first output as input for 2nd step as assistant:
		{ "role": "assistant", "content": json.dumps({"step": "analyse", "content": "the user is asking for an arithmetic operation that involves both addition and multiplication, so i need to follow the order of operations."})  },
	]
)
```

output:

```shell
{
	"step" : "think",
	"content" : "In order of operation,..."
}

```

Third step:

```python
result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"}, # expect the output in JSON
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is 3 + 4 * 5" },
		{ "role": "assistant", "content": json.dumps({"step": "analyse", "content": "the user is asking for an arithmetic operation that involves both addition and multiplication, so i need to follow the order of operations."})  },
		# passing the second output as input for 3rd step as assistant:
        { "role": "assistant", "content": json.dumps({"step": "think", "content": "In order of operations, multiplication should be performed before addition. Therefore, I should first multiply 4 by 5."}) },
	]
)
```

Output:

```shell
{
	"step": "think",
	"content": "Calculate the multiplication: 4 * 5 = 20."
}
```

Fourth step:

```python
result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"}, # expect the output in JSON
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is 3 + 4 * 5" },
		{ "role": "assistant", "content": json.dumps({"step": "analyse", "content": "the user is asking for an arithmetic operation that involves both addition and multiplication, so i need to follow the order of operations."})  },
        { "role": "assistant", "content": json.dumps({"step": "think", "content": "In order of operations, multiplication should be performed before addition. Therefore, I should first multiply 4 by 5."}) },
		# passing the third output as input for 4th step as assistant:
        { "role": "assistant", "content": json.dumps({"step": "think", "content": "Calculate the multiplication: 4 * 5 = 20."}) },
	]
)
```

Output:

```shell
{
	"step": "think",
	"content": "Next, I need to add the result of the multiplication (20) to the number 3."
}
```

Fifth step:

```python
result = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"}, # expect the output in JSON
    messages=[
        { "role": "system", "content": system_prompt },
        { "role": "user", "content": "what is 3 + 4 * 5" },
		{ "role": "assistant", "content": json.dumps({"step": "analyse", "content": "the user is asking for an arithmetic operation that involves both addition and multiplication, so i need to follow the order of operations."})  },
        { "role": "assistant", "content": json.dumps({"step": "think", "content": "In order of operations, multiplication should be performed before addition. Therefore, I should first multiply 4 by 5."}) },
        { "role": "assistant", "content": json.dumps({"step": "think", "content": "Calculate the multiplication: 4 * 5 = 20."}) },
		# passing the fourth output as input for 5th step as assistant:
        { "role": "assistant", "content": json.dumps({"step": "think", "content": "Next, I need to add the result of the multiplication (20) to the number 3."}) }
	]
)
```

Output:

```shell
{
	"step" : "output",
	"content" : "3 + 20 = 23"
}
```

> [!warning]
> These steps are supposed to be automated.

Automated flow:

```python
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

For the given user input, analyse the input and break down the problem step by step.
Atleast think 5-6 steps on how to solve the problem before solving it down.

The steps are you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

Rules:
1. Follow the strict JSON output as per Output schema.
2. Always perform one step at a time and wait for next input
3. Carefully analyse the user query

Output Format:
{{ step: "string", content: "string" }}

Example:
Input: What is 2 + 2.
Output: {{ step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }}
Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
Output: {{ step: "output", content: "4" }}
Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}

"""

messages = [
    { "role": "system", "content": system_prompt },
]


query = input("> ")
messages.append({ "role": "user", "content": query })


while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({ "role": "assistant", "content": json.dumps(parsed_response) })

    if parsed_response.get("step") != "output":
        print(f"🧠: {parsed_response.get("content")}")
        continue

    print(f"🤖: {parsed_response.get("content")}")
    break
```

Here, each JSON response is parsed to an object and appended in `messages`.

4. Self-Consistency Prompting: The model generates multiple responses and selects the most consistent or common answer.

Example user prompt: what is greater? 9.8 or 9.11

5. Instruction Prompting: The model is explicitly instructed to follow a particular format or guideline.

6. Direct Answer Prompting: The model is asked to give a concise and direct response without explanation.

7. Persona-based Prompting: The model is instructed to respond as if it were a particular character or professional.

8. Role-Playing Prompting: The model assumes a specific role and interacts accordingly.
   We assign role to the bot to set the context.
   Example system prompt: "your are an AI coding assistant who is expert in teaching how to code."

> [!tip]
> We can also mix multiple prompting techniques. Example: COT + Persona + Role.

9. Contextual Prompting: The prompt includes background information to improve response quality.

10. Multimodal Prompting: The model is given a combination of text, images, or other modalities to generate a response.

> [!NOTE] > **Contextual and multimodal (image, videos etc.) prompting** is complicated. This is where you need orchestration vector database, graph database etc.

---
