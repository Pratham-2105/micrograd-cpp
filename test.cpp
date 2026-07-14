#include "nn.hpp"
#include <iostream>
#include <memory>
#include <vector>

// Regression checks for the engine, one scoped block per build step.
// Each block is self-contained ({ }) so reused names don't collide.
// Run with:  make test
int main() {

  // --- graph recording: c = a * b remembers its children ---
  {
    auto a = std::make_shared<Value>(2.0);
    auto b = std::make_shared<Value>(3.0);
    auto c = mul(a, b);

    std::cout << "[graph] c->data              = " << c->data
              << "  (expect 6)\n";
    std::cout << "[graph] c->children[0]->data = " << c->children[0]->data
              << "  (expect 2)\n";
    std::cout << "[graph] c->children[1]->data = " << c->children[1]->data
              << "  (expect 3)\n";

    bool same = (c->children[0].get() == a.get());
    std::cout << "[graph] child0 same object as a? "
              << (same ? "true" : "false") << "  (expect true)\n";
  }

  // --- per-op backward closures: mul and add ---
  {
    auto a = std::make_shared<Value>(2.0);
    auto b = std::make_shared<Value>(3.0);
    auto c = mul(a, b);

    c->grad = 1.0;
    c->_backward();

    std::cout << "[mul bw] a->grad = " << a->grad << "  (expect 3)\n";
    std::cout << "[mul bw] b->grad = " << b->grad << "  (expect 2)\n";

    auto x = std::make_shared<Value>(2.0);
    auto y = std::make_shared<Value>(3.0);
    auto s = add(x, y);

    s->grad = 1.0;
    s->_backward();

    std::cout << "[add bw] x->grad = " << x->grad << "  (expect 1)\n";
    std::cout << "[add bw] y->grad = " << y->grad << "  (expect 1)\n";
  }

  // --- full automatic backward: topo sort + reverse walk ---
  {
    auto a = std::make_shared<Value>(2.0);
    auto b = std::make_shared<Value>(3.0);
    auto c = std::make_shared<Value>(10.0);
    auto e = mul(a, b); // 6
    auto L = add(e, c); // 16

    L->backward();

    std::cout << "[backward] L->data = " << L->data << "  (expect 16)\n";
    std::cout << "[backward] c->grad = " << c->grad << "  (expect 1)\n";
    std::cout << "[backward] a->grad = " << a->grad << "  (expect 3)\n";
    std::cout << "[backward] b->grad = " << b->grad << "  (expect 2)\n";
  }

  // --- shared-node accumulation: y = x + x, grad must SUM to 2 ---
  {
    auto x = std::make_shared<Value>(3.0);
    auto y = add(x, x);

    y->backward();

    std::cout << "[shared] x->grad = " << x->grad << "  (expect 2)\n";
  }

  // --- tanh activation ---
  {
    auto z = std::make_shared<Value>(0.0);
    auto t = tanh_(z);

    std::cout << "[tanh] t->data = " << t->data << "  (expect 0)\n";
    t->backward();
    std::cout << "[tanh] z->grad = " << z->grad << "  (expect 1)\n";
  }

  // --- Neuron: parameter count on construction ---
  {
    Neuron n(3);
    std::cout << "[neuron] w.size()    = " << n.w.size() << "  (expect 3)\n";
    std::cout << "[neuron] b->data     = " << n.b->data << "  (in [-1,1])\n";
    std::cout << "[neuron] w[0]->data  = " << n.w[0]->data << "  (in [-1,1])\n";
  }

  // --- Neuron: forward builds a graph backward can traverse ---
  {
    Neuron n(2);
    std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0),
                                             std::make_shared<Value>(-1.0)};

    auto out = n.forward(x);
    std::cout << "[neuron fw] out->data  = " << out->data << "  (in (-1,1))\n";

    out->backward();

    std::cout << "[neuron fw] w[0]->grad = " << n.w[0]->grad << "  (nonzero)\n";
    std::cout << "[neuron fw] b->grad    = " << n.b->grad << "\n";
  }

  // --- Layer: nout neurons on one input ---
  {
    Layer layer(2, 3);
    std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(0.5),
                                             std::make_shared<Value>(-0.5)};

    auto outs = layer.forward(x);
    std::cout << "[layer] outs.size() = " << outs.size() << "  (expect 3)\n";

    std::cout << "[layer] outs = ";
    for (auto &o : outs)
      std::cout << o->data << " ";
    std::cout << "\n";
  }

  // --- MLP: chained layers, gradient reaches the first layer ---
  {
    MLP net(3, {4, 4, 1});
    std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0),
                                             std::make_shared<Value>(-2.0),
                                             std::make_shared<Value>(0.5)};

    auto out = net.forward(x);

    std::cout << "[mlp] out.size()   = " << out.size() << "  (expect 1)\n";
    std::cout << "[mlp] out[0]->data = " << out[0]->data << "  (in (-1,1))\n";

    out[0]->backward();

    std::cout << "[mlp] layers[0].neurons[0].w[0]->grad = "
              << net.layers[0].neurons[0].w[0]->grad << "  (nonzero)\n";
  }

  // --- parameters() counts at each level ---
  {
    Neuron n(3);
    std::cout << "[params] neuron = " << n.parameters().size()
              << "  (expect 4)\n"; // 3 w + 1 b

    Layer layer(2, 3);
    std::cout << "[params] layer  = " << layer.parameters().size()
              << "  (expect 9)\n"; // 3 neurons * (2 w + 1 b) = 9

    MLP net(3, {4, 4, 1});
    std::cout << "[params] mlp    = " << net.parameters().size()
              << "  (expect 41)\n"; // 16 + 20 + 5
  }

  // --- sub op: a - b, with +1 / -1 local grads ---
  {
    auto p = std::make_shared<Value>(5.0);
    auto q = std::make_shared<Value>(3.0);
    auto r = sub(p, q);

    r->grad = 1.0;
    r->_backward();

    std::cout << "[sub] r->data=" << r->data << " p->grad=" << p->grad
              << " q->grad=" << q->grad << "  (expect 2 1 -1)\n";
  }

  // --- loss graph + one backward reaches the weights ---
  {
    std::vector<std::vector<Scalar>> xs = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
    std::vector<Scalar> ys = {1.0, -1.0, -1.0, 1.0};

    MLP net(3, {4, 4, 1});

    std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);

    for (u32 i = 0; i < xs.size(); ++i) {
      std::vector<std::shared_ptr<Value>> input;
      for (Scalar val : xs[i])
        input.push_back(std::make_shared<Value>(val));

      auto out = net.forward(input);
      auto pred = out[0];
      auto target = std::make_shared<Value>(ys[i]);
      auto diff = sub(pred, target);
      auto sq = mul(diff, diff);
      loss = add(loss, sq);
    }

    std::cout << "[loss] loss = " << loss->data << "  (finite, few-ish)\n";

    for (auto &p : net.parameters())
      p->grad = 0;
    loss->backward();

    auto params = net.parameters();
    std::cout << "[loss] param[0] grad = " << params[0]->grad
              << "  (nonzero)\n";
  }

  return 0;
}
