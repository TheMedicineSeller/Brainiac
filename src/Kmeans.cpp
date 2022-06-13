#include "Kmeans.hpp"

brainiac::LabelledKmeans::LabelledKmeans (int no_of_classes, int no_of_iterations) {
    this->epochs = no_of_iterations;
    this->K = no_of_classes;

    this->entropyEngine.seed(this->entropySource());

    this->centroids.reserve(no_of_classes);

    this->Clusters.resize(no_of_classes);
    this->dominant_label_maps.resize(no_of_classes);
}

double brainiac::LabelledKmeans::Distance (const std::vector<double> &imgfeature, const std::vector<double> &centroid) {
    uint32_t size = imgfeature.size();
    double distance = 0.0;
    for (uint32_t i = 0 ; i < size; i ++) {
        distance += (imgfeature[i] - centroid[i])*(imgfeature[i] - centroid[i]);
    }
    return sqrt(distance);
}

void brainiac::LabelledKmeans::UpdateClusterCentroid (brainiac::Imgdata* img, int cluster_idx) {

    std::vector<double> features = img->getFeatureVectorDouble();
    int cluster_size = this->Clusters.size();

    for (int i = 0; i < img->getFeatureSize(); i ++) {
        this->centroids[cluster_idx][i] = (this->centroids[cluster_idx][i] * 
                                           cluster_size + 
                                           features[i]) / (cluster_size + 1);
    }
    this->Clusters[cluster_idx].push_back(img);
}

void brainiac::LabelledKmeans::Train (const std::vector<brainiac::Imgdata*> &labelled_training_data) {
    printf("Training image data for %d epochs...\n", this->epochs);
    std::unordered_set<uint32_t> closedIndices;
    std::uniform_int_distribution<uint32_t> RandomIndexGenerator(0, labelled_training_data.size());

    while (this->centroids.size() < this->K) {
        uint32_t rand_idx = RandomIndexGenerator(this->entropyEngine);

        if (closedIndices.find(rand_idx) == closedIndices.end()) {
            std::vector<double> rand_pick = labelled_training_data[rand_idx]->getFeatureVectorDouble();
            this->centroids.emplace_back(rand_pick);
            closedIndices.insert(rand_idx);
        }
    }

    for (int _iteration = 0; _iteration < this->epochs; _iteration ++) {

        for (int i = 0; i < labelled_training_data.size(); i ++) {
            double distance, min_distance = std::numeric_limits<double>::max();
            int min_idx;
            std::vector<double> imgfeature = labelled_training_data[i]->getFeatureVectorDouble(); 

            for (int j = 0; j < this->centroids.size(); j ++) {
                distance = this->Distance(imgfeature, this->centroids[j]);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_idx = j;
                }
            }
            this->UpdateClusterCentroid(labelled_training_data[i], min_idx);
        }
        printf("Completed Iteration %d of training Kmeans classifier...\n", _iteration + 1);
        
        if (_iteration < this->epochs - 1) {
            this->Clusters.clear();
            this->Clusters.resize(this->K);
        }
    }
    
    for (int _cluster = 0; _cluster < this->K; _cluster ++) {

        for (int _img = 0; _img < this->Clusters[_cluster].size(); _img ++) {
            uint8_t _label = this->Clusters[_cluster][_img]->getClassLabel();

            if (this->dominant_label_maps[_cluster].find(_label) == this->dominant_label_maps[_cluster].end())
                this->dominant_label_maps[_cluster].insert({_label, 1});
            else
                this->dominant_label_maps[_cluster].at(_label) ++;
        }
    }
    printf("\nCompleted Training Kmeans classifier on training images...\n");
}

uint8_t brainiac::LabelledKmeans::Predict (brainiac::Imgdata* testimg) {
    double distance, min_distance = std::numeric_limits<double>::max();
    int min_idx;
    for (int i = 0; i < centroids.size(); i ++) {
        distance = this->Distance(testimg->getFeatureVectorDouble(), this->centroids[i]);
        if (distance < min_distance) {
            min_distance = distance;
            min_idx = i;
        }
    }
    this->UpdateClusterCentroid(testimg, min_idx);

    uint8_t max_label;
    int max_count = std::numeric_limits<int>::min();
    for (auto map_itr = this->dominant_label_maps[min_idx].begin();
              map_itr != this->dominant_label_maps[min_idx].end(); 
              ++ map_itr) {
        if (map_itr->second > max_count) {
            max_count = map_itr->second;
            max_label = map_itr->first;
        }
    }
    /*printf("Predicted : %hhu :: Expected %hhu...\n", max_label, testimg->getClassLabel());*/
    return max_label;
}

void brainiac::LabelledKmeans::ValidatePerformance (const std::vector<Imgdata*> &validationSet) {
    int hits = 0;
    uint8_t predicted_label;
    for (int i = 0; i < validationSet.size(); i ++) {
        predicted_label = this->Predict(validationSet[i]);
        if (predicted_label == validationSet[i]->getClassLabel())
            hits ++;
    }
    printf("Overall Performance of the Kmeans classifier over the given validation set : %lf%\n", (double)hits * 100 / validationSet.size() );
}