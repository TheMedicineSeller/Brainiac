#include "Imgdata.hpp"

brainiac::Imgdata::Imgdata (uint32_t size) {
    this->features.reserve(size);
    this->features_double.reserve(size);
}

void brainiac::Imgdata::setFeatureVector (const std::vector<uint8_t> &features) {
    this->features = features;
    for (int i = 0; i < features.size(); i ++) {
        this->features_double.emplace_back(static_cast<double>(features[i]));
    }
}

void brainiac::Imgdata::appendByte (uint8_t byte) {
    this->features.push_back(byte);
    this->features_double.push_back(static_cast<double>(byte));
}

void brainiac::Imgdata::setClassLabel (uint8_t label) {
    this->class_label = label;
}

uint8_t brainiac::Imgdata::getClassLabel () {
    return this->class_label;
}

uint32_t brainiac::Imgdata::getFeatureSize () {
    return this->features.size();
}

std::vector<uint8_t> brainiac::Imgdata::getFeatureVector () {
    return this->features;
}

std::vector<double> brainiac::Imgdata::getFeatureVectorDouble () {
    return this->features_double;
}