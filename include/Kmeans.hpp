#pragma once

#include "Imgdata.hpp"
#include <vector>
#include <random>
#include <unordered_set>
#include <limits>

namespace brainiac
{
    static inline double absolute (double x) {return (x >= 0) ? x : -x; }
    class LabelledKmeans {
        int K;
        int iteration_count;
        std::vector<std::vector<Imgdata*>> Clusters;
        std::vector<double>                centroids;

        std::random_device entropySource;
        std::mt19937       entropyEngine;

        double ImageMean(Imgdata* img);
        public:
            LabelledKmeans(int no_of_classes, int no_of_iterations=5);
            void Train(std::vector<Imgdata*> labelled_training_data);
            uint8_t Predict(Imgdata* img);
    };
}