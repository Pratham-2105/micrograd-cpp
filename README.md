# micrograd-cpp

A small scalar automatic differentiation (autograd) engine written from scratch in C++17,
with no external libraries. Each `Value` holds a single number and records the operation and
inputs that produced it, so many `Value`s form a graph of a computation. Calling `backward()`
on the final `Value` walks that graph in reverse and applies the chain rule to compute the
gradient of the output with respect to every input. This is the same mechanism (reverse-mode
autodiff) that underlies libraries like PyTorch, reduced to its simplest form.

On top of the engine sit small neural-network building blocks (Neuron, Layer, MLP) and a
training loop. Trained on a tiny toy dataset (4 examples), the MSE loss drops from ~6.1 to
~0.007 over 100 gradient-descent steps and the predictions converge on their targets — which
is what proves the engine is correct end to end.

This is a learning project — the point is to understand how autograd works by implementing it,
not to be fast or complete.

## Build

    make        # compile the training demo
    make run    # compile and run the training demo
    make test   # compile and run the engine checkpoints
    make clean  # remove the binaries

Requires a C++17 compiler.
