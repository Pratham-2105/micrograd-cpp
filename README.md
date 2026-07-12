# micrograd-cpp

A small scalar automatic differentiation (autograd) engine written from scratch in C++17,
with no external libraries. Each `Value` holds a single number and records the operation and
inputs that produced it, so many `Value`s form a graph of a computation. Calling `backward()`
on the final `Value` walks that graph in reverse and applies the chain rule to compute the
gradient of the output with respect to every input. This is the same mechanism (reverse-mode
autodiff) that underlies libraries like PyTorch, reduced to its simplest form.

This is a learning project — the point is to understand how autograd works by implementing it,
not to be fast or complete.

## Build

    make        # compile
    make run    # compile and run
    make clean  # remove the binary

Requires a C++17 compiler.
