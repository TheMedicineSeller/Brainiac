#include "Kmeans.hpp"

brainiac::LabelledKmeans::LabelledKmeans (int no_of_classes, int no_of_iterations) {
    this->iteration_count = no_of_iterations;
    this->K = no_of_classes;

    this->entropyEngine.seed(this->entropySource());

    this->Clusters.reserve(no_of_classes);
    this->centroids.resize(no_of_classes);
}

double brainiac::LabelledKmeans::ImageMean (brainiac::Imgdata* img) {
    double mean;
    std::vector<uint8_t> features = img->getFeatureVector();
    for (int p_ind = 0; p_ind < img->getFeatureSize(); p_ind ++) {
        mean += features[p_ind];
    }
    return (mean / img->getFeatureSize());
}

/*
double brainiac::LabelledKmeans::distance (brainiac::Imgdata* img1, brainiac::Imgdata* img2) {
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
}*/

void brainiac::LabelledKmeans::Train (std::vector<brainiac::Imgdata*> labelled_training_data) {
    if (Clusters.size() == 0) {

        std::unordered_set<uint32_t> closedIndices;
        std::uniform_int_distribution<std::mt19937::result_type> RandomIndexGenerator(0, labelled_training_data.size());

        while (this->centroids.size() < this->K) {
            uint32_t rand_idx = RandomIndexGenerator(this->entropyEngine);

            if (closedIndices.find(rand_idx) == closedIndices.end()) {
                double mean = this->ImageMean(labelled_training_data[rand_idx]);
                this->centroids.emplace_back(mean);
                closedIndices.insert(rand_idx);
            }
        }
    }

    for (int i = 0; i < labelled_training_data.size(); i ++) {
        double distance, min_distance = std::numeric_limits<double>::max();
        int min_idx;
        for (int j = 0; j < centroids.size(); j ++) {
            distance = absolute(centroids[i] - this->ImageMean(labelled_training_data[i]));
            if (distance < min_distance) {
                min_distance = distance;
                min_idx = j;
            }
        }
        this->Clusters[min_idx].push_back(labelled_training_data[i]);
        // Have to implement Iterations training
    }
}