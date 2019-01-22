// Registerer is a factory register mechanism that make class registration at
// compile time and allow generating an object by giving the name.
// Example:
//
//     class BaseClass {  // base class
//       ...
//     };
//     REGISTER_REGISTERER(BaseClass);
//     #define REGISTER_BASECLASS(name) REGISTER_CLASS(BaseClass, name)
//
//     class Sub1 : public BaseClass {
//       ...
//     };
//     REGISTER_BASE(Sub1);
//     class Sub2 : public BaseClass {
//       ...
//     };
//     REGISTER_BASE(Sub2);
//
// Note that REGISTER_BASE(sub1) should be put in cc file instead of h file,
// to avoid multi-declaration error when compile.
//
// Then you could get a new object of the sub class by:
//    Base *obj = BaseClassRegisterer::get_instance_by_name("Sub1");
//
// This is convenient when you need decide the class at runtime or by flag:
//    string name = "Sub1";
//    if (...)
//      name = "Sub2";
//    Base *obj = BaseClassRegisterer::get_instance_by_name(name);
//
// If there should be only one instance in the program by desgin,
// get_uniq_instance could be used:
//    Base *obj = BaseClassRegisterer::get_uniq_instance();
// If multi sub classes are registered, this method will cause a CHECK fail.

#ifndef REGISTERER_H
#define REGISTERER_H

#include <glog/logging.h>
#include <map>
#include <string>
#include <typeinfo>

#include "noncopyable.h"

namespace base {
namespace registerer {

// from library class Boost.Any
// see https://www.boost.org/doc/libs/1_61_0/doc/html/any.html for more detail
class Any {
public:
    Any() : _content(NULL) {}

    template <typename ValueType>
    Any(const ValueType &value)
        : _content(new Holder<ValueType>(value)) {}

    Any(const Any &other)  // rely on underlying copy constructor of ValueType, or NULL
        : _content(other._content ? other._content->clone() : NULL) {}

    ~Any() {
        delete _content;  // invoke underlying deconstructor
    }

    const std::type_info &type_info() const {
        return _content ? _content->type_info() : typeid(void);
    }

    template <typename ValueType>
    ValueType *to_ptr() const {
        // convert the "_content" into a ptr to its derived class "Holder" with ValueType (performed with check)
        // then get the address of its actual content (_held)
        return this->type_info() == typeid(ValueType) ? &static_cast<Holder<ValueType> *>(_content)->_held : NULL;
    }

    template <typename ValueType>
    ValueType *to_ptr_unsafe() const {
        // no typeid() expression to speed up, hence unsafe for no type check
        // (acquiring typeid under polymorphism needs virtual table look-up)
        return &static_cast<Holder<ValueType> *>(_content)->_held;
    }

    template <typename ValueType>
    ValueType any_cast() const {
        const ValueType *rst = this->to_ptr<ValueType>();  // try converting to a ValueType* type ptr
        if (rst) {
            return *rst;
        }
        throw std::bad_cast();
    }

    template <typename ValueType>
    ValueType any_cast_unsafe() const {  // speed up using to_ptr_unsafe()
        return _content ? *(this->to_ptr_unsafe<ValueType>()) : NULL;
    }

private:
    class PlaceHolder {
    public:
        virtual ~PlaceHolder() {}  // this makes the type polymorphic
        virtual PlaceHolder *clone() const = 0;
        virtual const std::type_info &type_info() const = 0;
    };

    template <typename ValueType>
    class Holder : public PlaceHolder {  // inheritance with template
    public:
        explicit Holder(const ValueType &value) : _held(value) {}
        virtual ~Holder() {}

        virtual PlaceHolder *clone() const {
            return new Holder(_held);
        }

        virtual const std::type_info &type_info() const {
            return typeid(ValueType);
        }

        ValueType _held;
    };

    PlaceHolder *_content;  // holding a polymorphic (generic) type
};

class ObjectFactory {
public:
    ObjectFactory() {}
    virtual ~ObjectFactory() {}
    virtual Any new_instance() {
        return Any();
    }

private:
    DISALLOW_COPY_AND_ASSIGN(ObjectFactory);
};

using FactoryMap = std::map<std::string, ObjectFactory *>;  // derived class name -> actual factory
using BaseClassMap = std::map<std::string, FactoryMap>;     // base_class name -> factory map
BaseClassMap &global_factory_map();

bool get_registered_classes(
    const std::string &base_class_name,
    std::vector<std::string> *registered_derived_classes_names);

}  // namespace registerer
}  // namespace base

#define REFISTER_NAMESPACE ::base::registerer

// create a class, able to search through the map on demand.
// specifically, used to get the factory map for the specified base_class, and .find( factory for derived class ) in it.
#define REGISTER_REGISTERER(base_class)                                                                                             \
    class base_class##Registerer {                                                                                                  \
        using Any = REFISTER_NAMESPACE::Any;                                                                                        \
        using FactoryMap = REFISTER_NAMESPACE::FactoryMap;                                                                          \
                                                                                                                                    \
    public:                                                                                                                         \
        static base_class *get_instance_by_name(const ::std::string &name) {                                                        \
            FactoryMap &map = REFISTER_NAMESPACE::global_factory_map()[#base_class]; /** insert new element in map if not found **/ \
            FactoryMap::iterator iter = map.find(name);                              /** find subclass by name **/                  \
            if (iter == map.end()) {                                                                                                \
                for (auto c : map) {                                                                                                \
                    LOG(ERROR) << "Instance:" << c.first;                                                                           \
                }                                                                                                                   \
                LOG(ERROR) << "Get instance " << name << " failed.";                                                                \
                return NULL;                                                                                                        \
            }                                                                                                                       \
            Any object = iter->second->new_instance(); /** virtual function call, derived ObjectFactory returns from "new" **/      \
            return object.any_cast<base_class *>();    /** cast back to a base_class* type ptr **/                                  \
        }                                                                                                                           \
        static std::vector<base_class *> get_all_instances() {                                                                      \
            std::vector<base_class *> instances;                                                                                    \
            FactoryMap &map = REFISTER_NAMESPACE::global_factory_map()[#base_class];                                                \
            instances.reserve(map.size());                                                                                          \
            for (auto item : map) {                                                                                                 \
                Any object = item.second->new_instance();                                                                           \
                instances.push_back(object.any_cast<base_class *>()); /** a vector of base_class* type ptr **/                      \
            }                                                                                                                       \
            return instances;                                                                                                       \
        }                                                                                                                           \
        static const ::std::string get_uniq_instance_name() {                                                                       \
            /** the selected FactpryMap contains only 1 derived class factory **/                                                   \
            FactoryMap &map = REFISTER_NAMESPACE::global_factory_map()[#base_class];                                                \
            CHECK_EQ(map.size(), 1) << map.size();                                                                                  \
            return map.begin()->first;                                                                                              \
        }                                                                                                                           \
        static base_class *get_uniq_instance() {                                                                                    \
            FactoryMap &map = REFISTER_NAMESPACE::global_factory_map()[#base_class];                                                \
            CHECK_EQ(map.size(), 1) << map.size();                                                                                  \
            Any object = map.begin()->second->new_instance();                                                                       \
            return object.any_cast<base_class *>();                                                                                 \
        }                                                                                                                           \
        static bool is_valid(const ::std::string &name) {                                                                           \
            FactoryMap &map = REFISTER_NAMESPACE::global_factory_map()[#base_class];                                                \
            return map.find(name) != map.end(); /** chk if base_class found **/                                                     \
        }                                                                                                                           \
    };

// define a class inheriting ObjectFactory, overriding new_instance() to instantiate class called "name"
// specifically, put a factory producing name* type pointer (via new) into the FactoryMap for base_class "clazz"
#define REGISTER_CLASS(clazz, name)                                                                                                \
    namespace {                                                                                                                    \
    using FactoryMap = REFISTER_NAMESPACE::FactoryMap;                                                                             \
    using Any = REFISTER_NAMESPACE::Any;                                                                                           \
                                                                                                                                   \
    class ObjectFactory##name : public REFISTER_NAMESPACE::ObjectFactory {                                                         \
    public:                                                                                                                        \
        virtual ~ObjectFactory##name() {}                                                                                          \
        virtual Any new_instance() {                                                                                               \
            /** implicit instantiation template constructor Any<name*> (new name()) **/                                            \
            return Any(new name()); /** thus holding a name* type ptr **/                                                          \
        }                                                                                                                          \
    };                                                                                                                             \
                                                                                                                                   \
    __attribute__((constructor)) void register_factory_##name() {           /** invoked before main() entered **/                  \
        FactoryMap &map = REFISTER_NAMESPACE::global_factory_map()[#clazz]; /** get / insert the base class (clazz) in the map **/ \
        if (map.find(#name) == map.end())                                   /** find the derived class (name) **/                  \
            map[#name] = new ObjectFactory##name();                         /** insert itself if not found (thus registered) **/   \
    }                                                                                                                              \
    }

#endif  // REGISTERER_H
