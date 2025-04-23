import json

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

client = OpenAI()


# zero shot prompting - the model is given a direct question or task without prior example
def zero_shot_prompting():
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "What is greater? 9.8 or 9.11"}],
    )
    print(result.choices[0].message.content)  # 9.8 is greater than 9.11.


# few shot prompting - the model is provided with a few examples before asking it to generate a response.
def few_shot_prompting():
    system_prompt = """
    You are an AI assistant. You will only reply to mathmatical questions. You will not reply to any of the non mathmatical questions.

    Example: 
    Input: what is mamals?
    Output: You are only supposed to question me math questions.

    Example: 
    Input: what is the result of 2*2?
    Output: The result of 2*2 is 4.
    """

    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "What is a mamal!"},
            # {"role": "user", "content": "What is difference between LCM and HCF"},
        ],
    )

    print(result.choices[0].message.content)


# COT - break down reasoning step by step before arriving at an answer
def chain_of_thought():
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
    Output: {{ step: "analyse", content: "Alright! The user is interested in maths query and he is asking a basic arthermatic operation" }}
    Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
    Output: {{ step: "output", content: "4" }}
    Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
    Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}
    """

    result = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "what is the result of 3*4-1"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "analyse",
                        "content": "The user wants to solve a mathematical expression involving multiplication and subtraction.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "think",
                        "content": "According to the order of operations (PEMDAS/BODMAS), I should first perform the multiplication before the subtraction.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "think",
                        "content": "The expression is 3*4-1. First, calculate the multiplication 3*4.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "think",
                        "content": "The expression is 3*4-1. First, calculate the multiplication 3*4.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {"step": "output", "content": "The result of 3*4 is 12."}
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "think",
                        "content": "Subtract 1 from the result of the multiplication, which is 12, to complete the expression 3*4-1.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "output",
                        "content": "The result of 3*4-1 is 12 - 1, which equals 11.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "validate",
                        "content": "The calculation steps followed the order of operations: first 3*4 = 12, then 12 - 1 = 11, which seems correct.",
                    }
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "step": "result",
                        "content": "The final result of the expression 3*4-1 is 11.",
                    }
                ),
            },
        ],
    )
    print(result.choices[0].message.content)


def chain_of_thought_automated():
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
    Output: {{ step: "analyse", content: "Alright! The user is interested in maths query and he is asking a basic arthermatic operation" }}
    Output: {{ step: "think", content: "To perform the addition i must go from left to right and add all the operands" }}
    Output: {{ step: "output", content: "4" }}
    Output: {{ step: "validate", content: "seems like 4 is correct ans for 2 + 2" }}
    Output: {{ step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }}
    """

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
    ]
    user_query = input("> ")
    messages.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gpt-4o", response_format={"type": "json_object"}, messages=messages
        )

        parsed_response = json.loads(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

        if parsed_response.get("step") != "result":
            print(f"🧠: {parsed_response.get("content")}")
            continue

        print(f"🤖: {parsed_response.get("content")}")
        break


# zero_shot_prompting()
# few_shot_prompting()
# chain_of_thought()
chain_of_thought_automated()
