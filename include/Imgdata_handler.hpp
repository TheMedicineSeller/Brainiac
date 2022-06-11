#pragma once

#include "Imgdata.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <map>

namespace brainiac {
    class Imgdata_handler {
        std::vector<Imgdata*> ImageArray;
        std::vector<Imgdata*> TrainArray;
        std::vector<Imgdata*> TestArray;
        std::vector<Imgdata*> ValidationArray;
        
        std::map<uint8_t, int> class_count_map;
        double TRAIN_RATIO;
        double TEST_RATIO;
        std::random_device entropySource;
        // void fillRandom(uint16_t count, std::unordered_set<int> &closed_set, int arrayclass);
        void updateClassCount();
        uint32_t to_little_endian(const uint8_t* bytes);

        public:
            Imgdata_handler(const char* featuresPath, const char* labelsPath , double train_percentage=0.7, double test_percentage=0.2);
            ~Imgdata_handler();

            void threeway_split_data();
            std::vector<Imgdata*> getTrainSet();
            std::vector<Imgdata*> getTestSet();
            std::vector<Imgdata*> getValSet();

            int classCount ();
    };
}