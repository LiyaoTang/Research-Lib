#include "randforest.hpp"
// #include "models/rsds-sk-randforest.h"

namespace models {
namespace rsds {
namespace carpoint {

/** 
 * definition for randforest carpoint
 * **/
CarPoint::CarPoint() : base_template::RandForest<std::vector<std::vector<double>>, std::vector<double>>(2, 8, 80) {}
CarPoint::~CarPoint() {}

void CarPoint::collect_pred(const std::vector<double> &input, std::vector<double> &classes) {
    classes.clear();
    classes.resize(class_num, 0);
    // vector to cnt pred from each tree
    // classes[predict_0(input)]++;
    // classes[predict_1(input)]++;
    // classes[predict_2(input)]++;
    // classes[predict_3(input)]++;
    // classes[predict_4(input)]++;
    // classes[predict_5(input)]++;
    // classes[predict_6(input)]++;
    // classes[predict_7(input)]++;
    // classes[predict_8(input)]++;
    // classes[predict_9(input)]++;
    // classes[predict_10(input)]++;
    // classes[predict_11(input)]++;
    // classes[predict_12(input)]++;
    // classes[predict_13(input)]++;
    // classes[predict_14(input)]++;
    // classes[predict_15(input)]++;
    // classes[predict_16(input)]++;
    // classes[predict_17(input)]++;
    // classes[predict_18(input)]++;
    // classes[predict_19(input)]++;
    // classes[predict_20(input)]++;
    // classes[predict_21(input)]++;
    // classes[predict_22(input)]++;
    // classes[predict_23(input)]++;
    // classes[predict_24(input)]++;
    // classes[predict_25(input)]++;
    // classes[predict_26(input)]++;
    // classes[predict_27(input)]++;
    // classes[predict_28(input)]++;
    // classes[predict_29(input)]++;
    // classes[predict_30(input)]++;
    // classes[predict_31(input)]++;
    // classes[predict_32(input)]++;
    // classes[predict_33(input)]++;
    // classes[predict_34(input)]++;
    // classes[predict_35(input)]++;
    // classes[predict_36(input)]++;
    // classes[predict_37(input)]++;
    // classes[predict_38(input)]++;
    // classes[predict_39(input)]++;
    // classes[predict_40(input)]++;
    // classes[predict_41(input)]++;
    // classes[predict_42(input)]++;
    // classes[predict_43(input)]++;
    // classes[predict_44(input)]++;
    // classes[predict_45(input)]++;
    // classes[predict_46(input)]++;
    // classes[predict_47(input)]++;
    // classes[predict_48(input)]++;
    // classes[predict_49(input)]++;
    // classes[predict_50(input)]++;
    // classes[predict_51(input)]++;
    // classes[predict_52(input)]++;
    // classes[predict_53(input)]++;
    // classes[predict_54(input)]++;
    // classes[predict_55(input)]++;
    // classes[predict_56(input)]++;
    // classes[predict_57(input)]++;
    // classes[predict_58(input)]++;
    // classes[predict_59(input)]++;
    // classes[predict_60(input)]++;
    // classes[predict_61(input)]++;
    // classes[predict_62(input)]++;
    // classes[predict_63(input)]++;
    // classes[predict_64(input)]++;
    // classes[predict_65(input)]++;
    // classes[predict_66(input)]++;
    // classes[predict_67(input)]++;
    // classes[predict_68(input)]++;
    // classes[predict_69(input)]++;
    // classes[predict_70(input)]++;
    // classes[predict_71(input)]++;
    // classes[predict_72(input)]++;
    // classes[predict_73(input)]++;
    // classes[predict_74(input)]++;
    // classes[predict_75(input)]++;
    // classes[predict_76(input)]++;
    // classes[predict_77(input)]++;
    // classes[predict_78(input)]++;
    // classes[predict_79(input)]++;
    return;
}

}  // namespace carpoint
}  // namespace rsds
}  // namepsace models