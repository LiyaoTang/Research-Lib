#include "registerer.h"

namespace base {
namespace registerer {

BaseClassMap& global_factory_map() {
    static BaseClassMap factory_map;
    return factory_map;
}

bool get_registered_classes(
    const std::string& base_class_name,
    std::vector<std::string>* registered_derived_classes_names) {
    CHECK_NOTNULL(registered_derived_classes_names);
    BaseClassMap& map = global_factory_map();
    auto iter = map.find(base_class_name);
    if (iter == map.end()) {
        LOG(ERROR) << "class not registered:" << base_class_name;
        return false;
    }
    for (auto pair : iter->second) {
        registered_derived_classes_names->push_back(pair.first);
    }
    return true;
}

}  // namespace registerer
}  // namespace base
