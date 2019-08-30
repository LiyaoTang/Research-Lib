#pragma once
#include <torch/script.h>
#include <opencv2/core.hpp>
#include "unit_base/unit_test_base.hpp"

namespace test{

class TorchTest : public UnitTestBase {
public:
    void test() {
        std::cout << "torch test" << std::endl;
        torch::jit::script::Module module;
        try {
            std::cout << "loading torch model" << std::endl;
            // module = torch::jit::load(); // deserialize the ScriptModule from a file
        } catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
        }
        std::cout << "ok\n";
    }
};

}