#pragma once
#include "torch_test.h"
#include "unit_base/unit_test_build.h"

namespace test {
class UnitTestEngine {
public:
    void test() {
        create();
        _unit_test_build.test();
    }

private:
    void create() {
        std::shared_ptr<UnitTestBase> unit_test_base = std::make_shared<BboxTest>();
        _unit_test_build.set_unit_test(unit_test_base);
    }

private:
    UnitTestBuild _unit_test_build;
};
}
