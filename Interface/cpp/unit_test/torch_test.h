#pragma once
#include <torch/script.h>
#include <opencv2/core.hpp>
#include "unit_base/unit_test_base.h"

namespace test{

class TorchTest : public UnitTestBase {
public:
    void test() {
        std::cout << "torch test" << std::endl;
        torch::jit::script::Module module;
        try {
            module = torch::jit::load(argv[1]); // deserialize the ScriptModule from a file
        } catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            return -1;
        }
        std::cout << "ok\n";
    }
};

}