#pragma once

#include <vector>

namespace brainiac {
    class Csvdata {
        std::vector<double> features;
        int label;
        std::vector<int> OneHotEncodedLabel;
        static int num_classes;
        public:
            void SetFeatures(const std::vector<double> &_features) {
                this->features = _features;
            }
            void AppendtoFeatures(double parameter) {
                this->features.push_back(parameter);
            }
            void SetLabel(int _label) {
                this->label = _label;
            }
            void EncodeLabel(int num_classes) {
                this->OneHotEncodedLabel.assign(num_classes, 0);
                this->OneHotEncodedLabel[this->label] = 1;
            }

            std::vector<double> getFeatures() {
                return this->features;
            }
            int getClassLabel() {
                return this->label;
            }
    };
}