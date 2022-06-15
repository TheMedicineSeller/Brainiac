#include "KNN.hpp"

brainiac::KNN::KNN (int _K) {
    this->K = _K;
}

double brainiac::KNN::euclidean_distance(brainiac::Imgdata* img1, brainiac::Imgdata* img2) {
    uint32_t size = img1->getFeatureSize();
    if (size != img2->getFeatureSize()) {
        printf("ERROR::Imagedistance : Unequal image sizes cannot be compared");
        exit(EXIT_FAILURE);
    }
    double distance = 0.0;
    std::vector<uint8_t> feature1 = img1->getFeatureVector(),
                         feature2 = img2->getFeatureVector();
    for (uint32_t i = 0 ; i < size; i ++) {
        distance += (feature1[i] - feature2[i])*(feature1[i] - feature2[i]);
    }
    return sqrt(distance);
}

double brainiac::KNN::manhattan_distance(brainiac::Imgdata* img1, brainiac::Imgdata* img2) {
    if (img1->getFeatureSize() != img2->getFeatureSize()) {
        printf("ERROR::Imagedistance : Unequal image sizes cannot be compared");
        exit(EXIT_FAILURE);
    }
    double distance = 0.0;
    std::vector<uint8_t> feature1 = img1->getFeatureVector(),
                         feature2 = img2->getFeatureVector();
    for (uint32_t i = 0; i < img1->getFeatureSize(); i ++) {
        distance += abs(feature1[i] - feature2[i]);
    }
    return distance;
}

void brainiac::KNN::LoadTraindata (std::vector<brainiac::Imgdata*> trainImages) {
    this->training_data = trainImages;
}

void brainiac::KNN::AppendTraindata (std::vector<brainiac::Imgdata*> new_train_data) {
    this->training_data.reserve(this->training_data.size() + new_train_data.size());
    this->training_data.insert(this->training_data.end(), new_train_data.begin(), new_train_data.end());
}

uint8_t brainiac::KNN::Predict (brainiac::Imgdata* img, bool euclidean) {

    std::vector<std::pair<uint8_t, double>> distanceLabel_pairs;
    distanceLabel_pairs.reserve(this->training_data.size());
    double dist;
    for (auto train_instance : this->training_data) {
        if (euclidean)
            dist = this->euclidean_distance(img, train_instance);
        else
            dist = this->manhattan_distance(img, train_instance);

        distanceLabel_pairs.emplace_back(std::make_pair(train_instance->getClassLabel(), dist));
    }

    std::partial_sort(distanceLabel_pairs.begin(), distanceLabel_pairs.begin() + this->K, distanceLabel_pairs.end(), 
                      [](std::pair<uint8_t, double> p1, std::pair<uint8_t, double> p2) -> bool
                      {
                        return p1.second < p2.second;
                      }
                     );
                     
    std::map<uint8_t, int> class_count;
    for (int i = 0; i < this->K; i ++) {
        if (class_count.find(distanceLabel_pairs[i].first) == class_count.end()) {
            class_count[distanceLabel_pairs[i].first] = 1;
            continue;
        }
        class_count[distanceLabel_pairs[i].first] ++;
    }

    uint8_t max_class;
    int     max_count = INT16_MIN;
    for (auto &keyvalue : class_count) {
        if (keyvalue.second > max_count) {
            max_count = keyvalue.second;
            max_class = keyvalue.first;
        }
    }
    return max_class;
}

double brainiac::KNN::ValidatePerformance (std::vector<Imgdata*> valimages) {
    int hits = 0;
    for (int itr = 0; itr < valimages.size(); itr ++) {
        if (valimages[itr]->getClassLabel() == this->Predict(valimages[itr]))
            hits ++;
        printf("Current performance of KNN classifier : %lf\n", hits * 100.0 / (itr+1));
    }
        
    return (hits * 1.0 / valimages.size());
}