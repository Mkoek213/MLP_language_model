# Name Generation Model - README

This project is an implementation of a character-level language model using an MLP (Multi-Layer Perceptron) to predict new names based on a dataset of English and Polish names. The model was inspired by [Andrej Karpathy's video](https://www.youtube.com/watch?v=TCH_1BHY58I) on building language models but modified for Polish name datasets and an updated architecture, achieving better performance.

The model is trained to predict names character by character, and after training, it is able to generate novel, human-like names. You can find the ready-to-use code in the `notebooks/model.ipynb` file.

## Performance

- **Train loss**: 1.98 (vs. Andrej's 2.12)
- **Validation loss**: 2.14 (vs. Andrej's 2.17)
- **Test loss**: 2.13

### Generated names from the trained model:
- mora
- kayah
- seel
- ndhayah
- remmas
- endra
- gradelynnelin
- shi
- jenne
- elieananar
- kateimilara
- nos
- davriah
- miel
- kin
- renlee
- jetta
- fiu
- zen
- dariyah

## Installation and Requirements

Run it using poetry:
```bash
poetry install
poetry shell
```


## Usage
To train and evaluate the model, follow the steps below:

### 1. Load and Process Data
First, read the dataset of names and build a vocabulary of characters.
```python
words = open('names.txt').read().splitlines()

# Build vocab of chars and mappings
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0  # Special token for end of name
itos = {i: s for s, i in stoi.items()}
block_size = 3  # Context length
```
**Explanation:**
The dataset of names is read from names.txt. We then build a character vocabulary (chars) and create mappings between characters and their indices (stoi for string-to-integer, itos for integer-to-string). A block size of 3 is used to define the context window for training.


### 2. Build the Dataset
The dataset is created by breaking down each name into sequences of characters and their subsequent characters as labels.
```python
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

```
**Explanation:**
The build_dataset function prepares the inputs (X) and labels (Y) for training, where each input is a sequence of characters, and the label is the next character. The model uses this information to predict subsequent characters.


### 3. Split Data into Train, Validation, and Test Sets
We split the dataset into training, validation, and test sets.
```python
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtrain, Ytrain = build_dataset(words[:n1])
Xval, Yval = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])
```
**Explanation:**
The data is shuffled and split into 80% training, 10% validation, and 10% test sets. This ensures that the model is evaluated on unseen data during training.


### 4. Initialize Parameters
Initialize the model parameters and embedding matrix.
```python
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((33, 10), generator=g)

W1 = torch.randn((30, 400), generator=g)
b1 = torch.randn(400, generator=g)

W2 = torch.randn((400, 33), generator=g)
b2 = torch.randn(33, generator=g)

parameters = [W1, b1, W2, b2]
```
**Explanation:**
An embedding matrix C and two layers of weights (W1, W2) and biases (b1, b2) are initialized with random values. These parameters will be learned during training.


### 5. Set Parameters to Require Gradients
Allow model parameters to be updated during backpropagation.
```python
for p in parameters:
    p.requires_grad = True
```
**Explanation:**
We enable gradient tracking for each parameter, allowing the model to update them during training.


### 6. Train the Model
The training loop performs mini-batch gradient descent with learning rate decay to optimize the model.
```python
lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
stepi = []

for i in range(500000):
    ix = torch.randint(0, Xtrain.shape[0], (128,))

    emb = C[Xtrain[ix]]
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytrain[ix])

    for p in parameters:
        p.grad = None
    loss.backward()

    if i < 100000:
        learning_rate = 10 ** (-0.7)
    elif i < 200000:
        learning_rate = (10 ** (-0.7))/10
    elif i < 300000:
        learning_rate = (10 ** (-0.7))/100
    else:
        learning_rate = (10 ** (-0.7))/1000

    for p in parameters:
        p.data += -learning_rate * p.grad

    stepi.append(i)
    lossi.append(loss.log10().item())

```
**Explanation:**
The model is trained for 500,000 iterations with mini-batch gradient descent. A custom learning rate schedule is applied, where the learning rate decreases as training progresses. Loss is logged at each step.

### 7. Plot Loss Curve
Visualize the training loss over iterations.
```python
plt.plot(stepi, lossi)
```
**Explanation:**
The plot shows how the model's loss decreases during training, providing insight into the training process.


### 8. Compute Final Losses
Evaluate the model on training, validation, and test sets.
```python
emb = C[Xtrain]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ytrain)
print(f"training loss: {loss}")

val_emb = C[Xval]
val_h = torch.tanh(val_emb.view(-1, 30) @ W1 + b1)
val_logits = val_h @ W2 + b2
val_loss = F.cross_entropy(val_logits, Yval)
print(f"validation loss: {val_loss}")

test_emb = C[Xtest]
test_h = torch.tanh(test_emb.view(-1, 30) @ W1 + b1)
test_logits = test_h @ W2 + b2
test_loss = F.cross_entropy(test_logits, Ytest)
print(f"test loss: {test_loss}")
```
**Explanation:**
This code computes and prints the final training, validation, and test losses, providing metrics for model performance on unseen data.


### 9. Visualize Character Embeddings
Plot a scatter plot of the learned character embeddings.
```python
plt.figure(figsize=(8, 8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha='center', va='center', color='white')
plt.grid('minor')
```
**Explanation:**
This visualizes the learned embeddings of each character in a 2D space, providing insights into how the model has organized different characters.


### 10. Generate New Names
Finally, the model is used to generate new names character by character.
```python
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    if len(out) > 2:
        print(''.join(itos[i] for i in out))
```
**Explanation:**
This loop generates new names by sampling characters from the model one at a time until an end-of-sequence token is reached. Each name is displayed after it's generated.
