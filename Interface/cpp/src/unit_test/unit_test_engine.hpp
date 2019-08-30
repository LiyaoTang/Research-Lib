#pragma once
#include "anchor_test.hpp"
// #include "bbox_test.hpp"
// #include "torch_test.hpp"
#include "unit_base/unit_test_build.hpp"

namespace test {
class UnitTestEngine {
public:
    void test() {
        create();
        _unit_test_build.test();
    }

private:
    void create() {
        std::shared_ptr<UnitTestBase> unit_test_base = std::make_shared<Anchor_Test>();
        _unit_test_build.set_unit_test(unit_test_base);
    }

private:
    UnitTestBuild _unit_test_build;
};
}
