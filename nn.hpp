#pragma once

#include "value.hpp"
#include <memory>
#include <random>
#include <vector>

struct Neuron {
  std::vector<std::shared_ptr<Value>> w;
  std::shared_ptr<Value> b;

  Neuron(i32 nin) {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<Scalar> dist(-1.0, 1.0);

    for (i32 i = 0; i < nin; ++i) {
      std::shared_ptr<Value> random_weight = std::make_shared<Value>(dist(rng));
      w.push_back(random_weight);
    }

    std::shared_ptr<Value> random_bias = std::make_shared<Value>(dist(rng));
    b = random_bias;
  }

  std::shared_ptr<Value> forward(std::vector<std::shared_ptr<Value>> x) {
    std::shared_ptr<Value> acc = b;
    for (u32 i = 0; i < w.size(); ++i) {
      acc = add(acc, mul(w[i], x[i]));
    }

    return tanh_(acc);
  }
};

struct Layer {
  std::vector<Neuron> neurons;

  Layer(i32 nin, i32 nout) {
    for (i32 i = 0; i < nout; ++i) {
      neurons.push_back(Neuron(nin));
    }
  }

  std::vector<std::shared_ptr<Value>>
  forward(std::vector<std::shared_ptr<Value>> x) {
    std::vector<std::shared_ptr<Value>> outs;

    for (Neuron &neuron : neurons) {
      outs.push_back(neuron.forward(x));
    }

    return outs;
  }
};
