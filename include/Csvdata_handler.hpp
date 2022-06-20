#pragma once

#include "Csvdata.hpp"
#include <fstream>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <stdio.h>

namespace brainiac {
    class Csvdata_handler {
        std::vector<Csvdata*> data;
        std::vector<Csvdata*> train_data;
        std::vector<Csvdata*> test_data;
        std::vector<Csvdata*> validation_data;

        std::unordered_map<std::string, int> class_count_map;
        std::random_device entropySource;

        public:
            Csvdata_handler(const char* filePath);
            void three_way_split(const double trainpercentage=0.7, const double testpercentage=0.2);
            std::vector<Csvdata*> getTestData();
            std::vector<Csvdata*> getValidationData();
            ~Csvdata_handler();
    };
}