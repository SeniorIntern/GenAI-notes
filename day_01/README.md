### Phase 1 - Tokenization

==A **token** is a chunk of text that the model reads as a single unit of meaning.==

User's input and encoding. The LLM divide these into tokens known as tokenization process (split in words).

For visualization reference - <https://tiktokenizer.vercel.app/>

"The cat sat on the mat" - this has 6 words or x letters. (_suppose_), thus 6 tokens.
Using a predefined dictionary, numeric representation is generated.
Example: The - 1, cat - 10, sat - 76

A token is not always a full word; it could be:

- A whole word (`apple`)
- Part of a word (`ap`, `ple`)
- A punctuation mark (`!`, `?`)
- A space character

For example:

- `"ChatGPT is cool!"` might be split into tokens like: `["Chat", "G", "PT", " is", " cool", "!"]`

Different models tokenize text differently, depending on the tokenizer they use. For example, OpenAI’s models often use a tokenizer called **Byte Pair Encoding (BPE)**.

> [!NOTE]
> Basically, Tokenization is representing user's input in numbers.

==Vocab Size - Total unique tokens. varies per modal.==
Example: 26 tokens for upper case, 26 for lower, 1 for space, 1 for dot, etc. So, vocab size = 26+26+1+1.

```python
import tiktoken

# To get the tokeniser corresponding to a specific model in the OpenAI API:
encoder = tiktoken.encoding_for_model("gpt-4o")
print("vocab size=", encoder.n_vocab)
# vocab size= 200019
```

```python
import tiktoken

# To get the tokeniser corresponding to a specific model in the OpenAI API:
encoder = tiktoken.encoding_for_model("gpt-4o")
print("vocab size=", encoder.n_vocab)
# vocab size= 200019

text = "The cat sat on the mat"
token = encoder.encode(text)
print(token)  # [976, 9059, 10139, 402, 290, 2450]
```

To decode you can use the `decode()` method.

```python
my_tokens = [976, 9059, 10139, 402, 290, 2450]
decoded = encoder.decode(my_tokens)

print(decoded)  # The cat sat on the mat
```

### Phase 2: Vector embedding are generated from the tokens

Vector embedding gets the relationship (semantic meaning) of words.

> [!NOTE]
> Vector embedding is basically semantic meaning/representation of user's input (meaning between words).

It happens in 3D space (X, Y, and Z).
For visualization reference - <https://projector.tensorflow.org/>

Generate vector embedding of text:

```python
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

text = "Eiffel Tower is in Paris and is a famous landmark, it is 324 meters tall"

# Generate text's vector embeddings
response = client.embeddings.create(input=text, model="text-embedding-3-small")

print("Vector embeddings", response.data[0].embedding)
```

Here, model size is like the dimension size of the vector embedding.
These vector embeddings can now be stored in vector database like pinecone db.

> [!tip]
> The cat sat the mat.
> That mat sat on he cat.
> These two will have same tokens but the meaning will be different.
>
> process: token -> vector embedding -> same semantic meaning

### Phase 3 - Positional Encoding

==The role of positional encoding is to provide the position of tokens. It works on vector embeddings to give context in positioning resulting in different encoding.==

Position of words matters. Thus, different tokens should be made for:
The cat sat the mat.
That mat sat on he cat.

> [!warning]
> Here, same tokens will be generated for these two sentences. The semantic meaning will be same.

In positional encoding, we use a formula to provide correct positions along with context of the position. We actually **provide numbers to later change their vector embeddings**.

### Phase 4 - Multi-headed attention mechanism

**self attention mechanism**
==Give chance to tokens to talk with each other to update itself based on the full sentence i.e. change vector embedding.==
You only have one head in self attention mechanism.

The bank river.
The ICICI bank

will change to:

The river bank.
The ICICI bank

- Self attention - how tokens interact with each other
- Multi Head Attention - basically do the task of single headed attention in parallel in multiple heads. It basically focuses on different-different aspect(what, who, when etc.) of tokens. It gets multiple context of how tokens interact with each other.

## Training, Inferencing, Linear, and Softmax

Training Phase
SCENARIO - For an input, an untrained modal gave a gibberish output. The expected output was "end" but it gave "asdfas".
Next Step:
We calculate loss based on the actual output and the model's output. Based on the loss, we do **_back propagation_** - This process is repeated using **_feed forward_** until we get an "end".
This consume a lot of GPU, thus it not feasible to do it daily.

Inferencing Phase - using the AI model
say user's input is - `<start> How are you? <end>`

from AI model side, the response word can be one of - I, A, B, C, D

The **_linear function_** gives probability to each response for next token. l - 90%, A - 49%, B - 3%, ..
The **_softmax_** will pick the one with highest probability. The softmax depends on the temperature set by the user or the default.
For example:

- "NDA" has a probability of 90
- "INDIA" has a probability of 10

Softmax will choose "NDA" because it has a higher probability.

Temperature is the creativity allowed to the AI model. high temperature - gets creative, accuracy decreases

> [!NOTE]
> Increasing the softmax/temperature will pick less probable next token.

The cycle would look like:

```txt
<start> How are you? <end>
I
// the data is then passed to input to go through all the steps
I (with space)
I am
```
