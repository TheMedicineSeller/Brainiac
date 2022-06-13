#pragma once

#include <vector>
#include <stdint.h>

namespace brainiac
{
    class Imgdata {
        std::vector<uint8_t> features;
        std::vector<double>  features_double;
        uint8_t class_label;
        
        public:
            Imgdata(uint32_t size);
            void setFeatureVector(const std::vector<uint8_t> &features);
            void appendByte(uint8_t byte);
            void setClassLabel (uint8_t label);

            std::vector<uint8_t> getFeatureVector();
            std::vector<double>  getFeatureVectorDouble();
            uint8_t getClassLabel();
            uint32_t getFeatureSize();
            // int getEnumClassLabel();
    };
}
