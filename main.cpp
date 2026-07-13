#include "nn.hpp"
#include <iostream>
#include <memory>
#include <vector>

int main() {
  /*
    Value a(2), b(3);
    Value c = a + b;
    Value d = a * b;

    std::cout << "c: " << c.data << " d: " << d.data << "\n";
  */

  /*
  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);

  auto c = mul(a, b);

  std::cout << "c->data            = " << c->data << "  (expect 6)\n";
  std::cout << "c->children[0]->data = " << c->children[0]->data
            << "  (expect 2)\n";
  std::cout << "c->children[1]->data = " << c->children[1]->data
            << "  (expect 3)\n";

  bool same = (c->children[0].get() == a.get());
  std::cout << "child0 is same object as a? " << (same ? "true" : "false")
            << "  (expect true)\n";

  */
  /*

  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);
  auto c = mul(a, b);

  c->grad = 1.0;
  c->_backward();

  std::cout << "a->grad = " << a->grad << "  (expect 3)\n";
  std::cout << "b->grad = " << b->grad << "  (expect 2)\n";

  auto x = std::make_shared<Value>(2.0);
  auto y = std::make_shared<Value>(3.0);
  auto s = add(x, y);

  s->grad = 1.0;
  s->_backward();

  std::cout << "x->grad = " << x->grad << "  (expect 1)\n";
  std::cout << "y->grad = " << y->grad << "  (expect 1)\n";
  */

  /*
  auto a = std::make_shared<Value>(2.0);
  auto b = std::make_shared<Value>(3.0);
  auto c = std::make_shared<Value>(10.0);
  auto e = mul(a, b); // 6
  auto L = add(e, c); // 16

  L->backward();

  std::cout << "L->data = " << L->data << "  (expect 16)\n";
  std::cout << "c->grad = " << c->grad << "  (expect 1)\n";
  std::cout << "a->grad = " << a->grad << "  (expect 3)\n";
  std::cout << "b->grad = " << b->grad << "  (expect 2)\n";

  auto x = std::make_shared<Value>(3.0);
  auto y = add(x, x);

  y->backward();

  std::cout << "x->grad = " << x->grad << "  (expect 2)\n";

  */

  /*
  auto z = std::make_shared<Value>(0.0);
  auto t = tanh_(z);

  std::cout << "t->data = " << t->data << "  (expect 0)\n";
  t->backward();
  std::cout << "z->grad = " << z->grad << "  (expect 1)\n";

  */

  /*
  Neuron n(3);
  std::cout << "w.size() = " << n.w.size() << "  (expect 3)\n";
  std::cout << "b->data  = " << n.b->data << "  (in [-1,1])\n";
  std::cout << "w[0]->data = " << n.w[0]->data << "  (in [-1,1])\n";
 */

  /*
  Neuron n(2);
  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0),
                                           std::make_shared<Value>(-1.0)};

  auto out = n.forward(x);
  std::cout << "out->data = " << out->data << "  (in (-1,1))\n";

  out->backward();

  std::cout << "w[0]->grad = " << n.w[0]->grad << "  (nonzero)\n";
  std::cout << "b->grad    = " << n.b->grad << "  (expect 1... ish)\n";
  */

  /*
  Layer layer(2, 3);
  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(0.5),
                                           std::make_shared<Value>(-0.5)};

  auto outs = layer.forward(x);
  std::cout << "outs.size() = " << outs.size() << "  (expect 3)\n";

  for (auto &o : outs)
    std::cout << o->data << " ";
  std::cout << "\n";
  */

  /*
  MLP net(3, {4, 4, 1});
  std::vector<std::shared_ptr<Value>> x = {std::make_shared<Value>(1.0),
                                           std::make_shared<Value>(-2.0),
                                           std::make_shared<Value>(0.5)};

  auto out = net.forward(x);

  std::cout << "out.size() = " << out.size() << "  (expect 1)\n";
  std::cout << "out[0]->data = " << out[0]->data << "  (in (-1,1))\n";

  out[0]->backward();

  std::cout << "net.layers[0].neurons[0].w[0]->grad = "
            << net.layers[0].neurons[0].w[0]->grad << "  (nonzero)\n";

 */

  /*
    Neuron n(3);
    std::cout << "neuron params = " << n.parameters().size()
              << "  (expect 4)\n"; // 3 w + 1 b

    Layer layer(2, 3);
    std::cout << "layer params = " << layer.parameters().size()
              << "  (expect 9)\n"; // 3 neurons * (2 w + 1 b) = 9

    MLP net(3, {4, 4, 1});
    std::cout << "mlp params = " << net.parameters().size()
              << "  (expect anything)\n";
  */

  /*
  auto p = std::make_shared<Value>(5.0);
  auto q = std::make_shared<Value>(3.0);
  auto r = sub(p, q);

  r->grad = 1.0;
  r->_backward();

  std::cout << r->data << " " << p->grad << " " << q->grad << "\n";
    */

  /*
  std::vector<std::vector<Scalar>> xs = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  std::vector<Scalar> ys = {1.0, -1.0, -1.0, 1.0};

  MLP net(3, {4, 4, 1});

  std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);

  for (u32 i = 0; i < xs.size(); ++i) {
    std::vector<std::shared_ptr<Value>> input;

    for (Scalar val : xs[i]) {
      input.push_back(std::make_shared<Value>(val));
    }

    auto out = net.forward(input);
    auto pred = out[0];

    auto target = std::make_shared<Value>(ys[i]);

    auto diff = sub(pred, target);
    auto sq = mul(diff, diff);

    loss = add(loss, sq);
  }

  std::cout << "loss = " << loss->data << '\n';

  for (auto &p : net.parameters())
    p->grad = 0;
  loss->backward();

  auto params = net.parameters();
  std::cout << "param[0] grad = " << params[0]->grad << "  (nonzero)\n";

  */

  std::vector<std::vector<Scalar>> xs = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  std::vector<Scalar> ys = {1.0, -1.0, -1.0, 1.0};

  MLP net(3, {4, 4, 1});

  for (int step = 0; step < 100; ++step) {
    std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);

    for (u32 i = 0; i < xs.size(); ++i) {
      std::vector<std::shared_ptr<Value>> input;

      for (Scalar val : xs[i]) {
        input.push_back(std::make_shared<Value>(val));
      }

      auto out = net.forward(input);
      auto pred = out[0];

      auto target = std::make_shared<Value>(ys[i]);

      auto diff = sub(pred, target);
      auto sq = mul(diff, diff);

      loss = add(loss, sq);
    }

    for (auto &p : net.parameters())
      p->grad = 0;
    loss->backward();

    for (auto &p : net.parameters()) {
      p->data += -0.05 * p->grad;
    }

    std::cout << "step " << step << " loss " << loss->data << "\n";
  }

  return 0;
}
