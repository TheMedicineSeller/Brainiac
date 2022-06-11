#include "Imgdata.hpp"

brainiac::Imgdata::Imgdata (uint32_t size) {
    this->features.reserve(size);
}

void brainiac::Imgdata::setFeatureVector (const std::vector<uint8_t> &features) {
    this->features = features;
}

void brainiac::Imgdata::appendByte (uint8_t byte) {
    this->features.push_back(byte);
}

void brainiac::Imgdata::setClassLabel (uint8_t label) {
    this->class_label = label;
}

uint8_t brainiac::Imgdata::getClassLabel () {
    return this->class_label;
}

// inline int brainiac::Imgdata::getEnumClassLabel () {
//     return reinterpret_cast<int> (this->class_label);
// }

uint32_t brainiac::Imgdata::getFeatureSize () {
    return this->features.size();
}

std::vector<uint8_t> brainiac::Imgdata::getFeatureVector () {
    return this->features;
}