#pragma once

#include "Imgdata.hpp"
#include <algorithm>
#include <map>
#include <cmath>
#include <stdio.h>

namespace brainiac
{
    class KNN {
        int K;
        std::vector<Imgdata*> training_data;
        std::vector<uint8_t> neighbour_class_buffer;
        double euclidean_distance(Imgdata* img1, Imgdata* img2);
        double manhattan_distance(Imgdata* img1, Imgdata* img2);
        // bool distanceComparator(const std::pair<uint8_t, double> &p1, const std::pair<uint8_t, double> &p2);
        public:
            KNN(int _K);
            void update_K(int _K) {
                this->K = _K;
            }
            // void TrainModel(std::vector<Imgdata*> trainImages);
            void LoadTraindata(std::vector<Imgdata*> trainImages);
            void AppendTraindata(std::vector<Imgdata*> new_training_data);
            uint8_t Predict(Imgdata* testimg, bool euclidean=true);
            double ValidatePerformance(std::vector<Imgdata*> valImages);
    };
}
