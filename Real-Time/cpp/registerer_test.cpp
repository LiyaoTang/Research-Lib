#include "registerer.h"
#include <gtest/gtest.h>

namespace base {
namespace registerer {

class BaseClass {
public:
    BaseClass() = default;
    ~BaseClass() = default;
    virtual std::string name() const {
        return "BaseClass1";
    }
};
REGISTER_REGISTERER(BaseClass);
#define REGISTER_TEST(name) REGISTER_CLASS(BaseClass, name)

class DerivedClass1 : BaseClass {
public:
    DerivedClass1() = default;
    ~DerivedClass1() = default;
    virtual std::string name() const {
        return "DerivedClass1";
    }
};
REGISTER_TEST(DerivedClass1);

class DerivedClass2 : BaseClass {
public:
    DerivedClass2() = default;
    ~DerivedClass2() = default;
    virtual std::string name() const {
        return "DerivedClass2";
    }
};
REGISTER_TEST(DerivedClass2);

TEST(RegistererTest, test) {
    BaseClass* ptr = nullptr;
    ptr = BaseClassRegisterer::get_instance_by_name("DerivedClass1");
    ASSERT_TRUE(ptr != nullptr);
    EXPECT_EQ(ptr->name(), "DerivedClass1");
    ptr = BaseClassRegisterer::get_instance_by_name("DerivedClass2");
    ASSERT_TRUE(ptr != nullptr);
    EXPECT_EQ(ptr->name(), "DerivedClass2");

    ptr = BaseClassRegisterer::get_instance_by_name("NotExists");
    ASSERT_TRUE(ptr == nullptr);

    std::vector<std::string> derived_classes;
    EXPECT_TRUE(get_registered_classes("BaseClass", &derived_classes));
    EXPECT_EQ(derived_classes.size(), 2u);
    EXPECT_TRUE(derived_classes[0] == "DerivedClass1" || derived_classes[0] == "DerivedClass2");
    EXPECT_TRUE(derived_classes[1] == "DerivedClass1" || derived_classes[1] == "DerivedClass2");
}

}  // namespace registerer
}  // namespace base
