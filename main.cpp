#include "nn.hpp"
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

int main() {
  std::vector<std::vector<Scalar>> xs = {
      {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};
  std::vector<Scalar> ys = {1.0, -1.0, -1.0, 1.0};

  MLP net(3, {4, 4, 1});

  std::ofstream lossfile("losses.txt");
  for (int step = 0; step < 100; ++step) {
    std::shared_ptr<Value> loss = std::make_shared<Value>(0.0);

    for (u32 i = 0; i < xs.size(); ++i) {
      std::vector<std::shared_ptr<Value>> input;

      for (Scalar val : xs[i])
        input.push_back(std::make_shared<Value>(val));

      auto out = net.forward(input);
      auto diff = sub(out[0], std::make_shared<Value>(ys[i]));

      loss = add(loss, mul(diff, diff));
    }

    for (auto &p : net.parameters())
      p->grad = 0;

    loss->backward();

    for (auto &p : net.parameters())
      p->data += -0.05 * p->grad;

    std::cout << "step " << step << " loss " << loss->data << "\n";
    lossfile << step << " " << loss->data << "\n";
  }

  return 0;
}
