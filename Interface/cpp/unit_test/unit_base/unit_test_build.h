#pragma once
#include "unit_test_base.h"

namespace test {
    class UnitTestBuild : public UnitTestBase {
    public:

        void set_unit_test(std::shared_ptr <UnitTestBase> unit_test) {
            _unit_test = unit_test;
        }

        void test() {
            _unit_test->test();
        }

    private:
        std::shared_ptr <UnitTestBase> _unit_test;
    };
}