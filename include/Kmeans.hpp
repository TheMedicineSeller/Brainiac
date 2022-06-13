#pragma once

#include "Imgdata.hpp"
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <limits>
#include <stdio.h>

namespace brainiac
{
    class LabelledKmeans {
        int K;
        int epochs;
        std::vector<std::vector<Imgdata*>> Clusters;
        std::vector<std::vector<double>>   centroids;
        std::vector<std::unordered_map<uint8_t, int>> dominant_label_maps;

        std::random_device entropySource;
        std::mt19937       entropyEngine;

        double Distance(const std::vector<double> &imgfeature, const std::vector<double> &centroid);
        void UpdateClusterCentroid(Imgdata* img, int cluster_idx);
        public:
            LabelledKmeans(int no_of_classes, int no_of_iterations=5);
            void Train(const std::vector<Imgdata*> &labelled_training_data);
            uint8_t Predict(Imgdata* testimg);
            void ValidatePerformance(const std::vector<Imgdata*> &validationSet);
    };
}