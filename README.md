# micrograd-cpp

A small scalar automatic differentiation engine written from scratch in C++. No libraries, no
frameworks — just `std::shared_ptr`, a graph of nodes, and the chain rule. It computes exact
gradients for any expression you build out of its `Value` type, and on top of it sits a tiny
neural network that trains itself with those gradients.

This is the follow-on to CNeural, and the contrast is the whole point. In CNeural I derived
backpropagation by hand and wrote it as fixed code for one specific network. micrograd flips
that: instead of hand-coding the gradients, it builds an engine that works out backpropagation
*automatically*, at runtime, for whatever expression you give it — the same idea that sits
underneath PyTorch and TensorFlow, reduced to its simplest form (one scalar per node).

The point wasn't speed or completeness. It was to understand automatic differentiation by
building it — every node, every local gradient rule, the topological sort, the backward pass,
written and checked by hand instead of imported.

## Results

The engine trains a small MLP (3 inputs -> 4 -> 4 -> 1) on Karpathy's toy dataset of 4
examples with binary targets, using mean-squared-error loss and plain gradient descent.
The loss over 100 steps:

    step 0    loss 6.0916
    step 5    loss 3.5675
    step 10   loss 1.0480
    step 15   loss 0.1879
    step 20   loss 0.0861
    step 40   loss 0.0235
    step 70   loss 0.0105
    step 99   loss 0.0067

The loss falls by roughly 900x. A final loss of 0.0067 spread across 4 examples means each
prediction sits within about 0.04 of its target — the network has learned the mapping. When
the loss drops like this and the predictions converge, the whole engine is proven correct end
to end: the forward pass built the graph, and the automatic backward pass computed every
gradient right.

## Running it

    make run    # compile and run the training demo
    make test   # compile and run the engine checkpoints
    make clean  # remove the binaries

or directly:

    g++ -std=c++17 -Wall -Wextra -O2 main.cpp -o micrograd && ./micrograd

Only a C++17 compiler is required. Nothing else.

## How it's put together

The code is split into a few files, each doing one job:

`value.hpp` — the engine. A `Value` struct that holds a number (`data`), its gradient
(`grad`), the child nodes it was built from, and a small closure (`_backward`) that knows how
to send gradient to those children. The operations — `add`, `mul`, `tanh_`, `sub` — each build
a new node, record its children, and attach the right backward rule. `backward()` does a
topological sort of the graph and walks it in reverse, calling each node's closure.

`nn.hpp` — the network. A `Neuron` (a weight `Value` per input, a bias, and a `tanh` on the
weighted sum), a `Layer` (a list of neurons run on the same input), and an `MLP` (layers
chained end to end). Each exposes `parameters()` so the training loop can reach every weight.

`main.cpp` — the training program. Builds the toy dataset, constructs the MLP, and runs the
loop: forward all examples, compute the loss, zero the gradients, `backward()`, then nudge
every parameter against its gradient. Repeat, watch the loss fall.

`test.cpp` — hand-checks for every piece of the engine, built up while writing it: graph
recording, each operation's backward rule, the full backward pass, gradient accumulation on a
shared node, and the network building blocks. Run with `make test`.

## How the learning works

Every operation on a `Value` builds a node that remembers its inputs and how it was made, so a
whole expression becomes a graph. `backward()` seeds the final node's gradient to 1, then walks
the graph in reverse (children after parents), and at each node applies the chain rule using a
local derivative rule specific to that operation:

| Operation | Local gradient rule                                        |
| --------- | ---------------------------------------------------------- |
| add       | each input gets the incoming gradient unchanged            |
| sub       | `a` gets `+grad`, `b` gets `-grad`                         |
| mul       | each input gets the *other* input's value times the gradient |
| tanh      | the input gets `(1 - tanh^2)` times the gradient           |

Gradients accumulate (`+=`) rather than overwrite, so when one node feeds several parts of an
expression, the contributions from every path sum correctly. Training is then just: build the
loss as one big `Value`, call `backward()` to fill in every weight's gradient, and step each
weight a little against its gradient. Do that a hundred times and the random starting weights
turn into something that fits the data.

## Notes

I built this one piece at a time, testing each part on small values I could check by hand
before moving on — the killer check is running `backward()` on a small expression and verifying
every gradient against a paper chain-rule computation. The rule throughout was to look up how a
mechanism works when I got stuck, but never to copy a finished solution; the point was to end
up understanding every line.

The design follows Andrej Karpathy's micrograd (from his *Neural Networks: Zero to Hero*
series), rebuilt in C++ from scratch. Where CNeural hand-coded backpropagation for a fixed
network, this automates it for any expression — the same progression the field itself took.
