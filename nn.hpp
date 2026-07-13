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

  std::vector<std::shared_ptr<Value>> parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (std::shared_ptr<Value> &weight : w) {
      params.push_back(weight);
    }
    params.push_back(b);

    return params;
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

  std::vector<std::shared_ptr<Value>> parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (auto &n : neurons) {
      auto np = n.parameters();
      params.insert(params.end(), np.begin(), np.end());
    }

    return params;
  }
};

struct MLP {
  std::vector<Layer> layers;

  MLP(i32 nin, std::vector<i32> nouts) {
    std::vector<i32> sizes;
    sizes.push_back(nin);

    for (i32 n : nouts)
      sizes.push_back(n);

    for (u32 i = 0; i < nouts.size(); ++i) {
      layers.push_back(Layer(sizes[i], sizes[i + 1]));
    }
  }

  std::vector<std::shared_ptr<Value>>
  forward(std::vector<std::shared_ptr<Value>> x) {

    for (Layer layer : layers) {
      x = layer.forward(x);
    }

    return x;
  }

  std::vector<std::shared_ptr<Value>> parameters() {
    std::vector<std::shared_ptr<Value>> params;

    for (auto &l : layers) {
      auto lp = l.parameters();
      params.insert(params.end(), lp.begin(), lp.end());
    }

    return params;
  }
};
