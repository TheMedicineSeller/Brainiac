#include <vector>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <chrono>
#include <iostream>

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
    };
}

brainiac::Imgdata::Imgdata (uint32_t size) {
    this->features.reserve(size);
    this->features_double.reserve(size);
}

void brainiac::Imgdata::setFeatureVector (const std::vector<uint8_t> &features) {
    this->features = features;
    for (int i = 0; i < features.size(); i ++) {
        this->features_double.emplace_back(static_cast<double>(features[i]));
    }
}

void brainiac::Imgdata::appendByte (uint8_t byte) {
    this->features.push_back(byte);
    this->features_double.push_back(static_cast<double>(byte));
}

void brainiac::Imgdata::setClassLabel (uint8_t label) {
    this->class_label = label;
}

uint8_t brainiac::Imgdata::getClassLabel () {
    return this->class_label;
}

// inline int brainiac::Imgdata::getEnumClassLabel () {
//     return reinterpret_cast<int> (this->class_label);
// }

uint32_t brainiac::Imgdata::getFeatureSize () {
    return this->features.size();
}

std::vector<uint8_t> brainiac::Imgdata::getFeatureVector () {
    return this->features;
}

std::vector<double> brainiac::Imgdata::getFeatureVectorDouble () {
    return this->features_double;
}

void brainiac::Imgdata_handler::updateClassCount () {
    uint16_t size;
    for (uint16_t i = 0; i < this->ImageArray.size(); i ++) {
        if (this->class_count_map.find(ImageArray[i]->getClassLabel()) != this->class_count_map.end())
            continue;
        this->class_count_map[ImageArray[i]->getClassLabel()] = size++;
    }
}

uint32_t brainiac::Imgdata_handler::to_little_endian (const uint8_t* bytes) {
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

brainiac::Imgdata_handler::Imgdata_handler (const char* featuresPath, const char* labelsPath, double TrainRatio, double TestRatio) 
: TRAIN_RATIO(TrainRatio), TEST_RATIO(TestRatio) {
    /*
    Specifically to handle MNIST image dataset format
    */
    uint32_t img_header[4];
    uint8_t buffer[4];

    FILE* features_fhandle = fopen(featuresPath, "rb");
    if (! features_fhandle) {
        perror("Could not open/find features file");
        exit(EXIT_FAILURE);
    }
    for (auto i : {0, 1, 2, 3})
        if (fread(buffer, sizeof(buffer), 1, features_fhandle))
            img_header[i] = this->to_little_endian(buffer);
    
    uint32_t img_size = img_header[2] * img_header[3];
    for (uint32_t i = 0; i < img_header[1]; i ++) {
        Imgdata* img = new Imgdata(img_size);
        /* Slightly More efficient in load time compared to vector & reserve but is dataset specific
        uint8_t pixels[784];
        */
        std::vector<uint8_t> pixels;
        pixels.reserve(img_size);
    
        if (! fread(&pixels[0], 1, sizeof(pixels), features_fhandle)) {
            perror("Error reading bytes from features file...\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0 ; i < img_size; i ++) {
            img->appendByte(pixels[i]);
        }
        this->ImageArray.emplace_back(img);
    }
    fclose(features_fhandle);
    printf("Successfully read and loaded %d image features...\n", img_header[1]);

    
    uint32_t label_header[2];
    uint8_t l_buffer[4];

    FILE* labels_fhandle = fopen(labelsPath, "rb");
    if (! labels_fhandle) {
        perror("Could not open/find labels file");
        exit(EXIT_FAILURE);
    }
    for (auto i : {0, 1})
        if (fread(l_buffer, sizeof(l_buffer), 1, labels_fhandle))
            label_header[i] = this->to_little_endian(l_buffer);

    /*uint8_t labels[60000];*/
    std::vector<uint8_t> labels;
    labels.reserve(img_header[1]);
    if (! fread(&labels[0], 1, sizeof(labels), labels_fhandle)) {
        perror("Error reading bytes from labels file...\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < img_header[1]; i ++) {
        this->ImageArray[i]->setClassLabel(labels[i]);
    }

    fclose(labels_fhandle);
    printf("Successfully read and loaded %d feature labels...\n", ImageArray.size());

    this->updateClassCount();
}

/*void brainiac::Imgdata_handler::fillRandom (uint16_t size, std::unordered_set<int> &closed_set, int arrayclass) {
    while (size > 0) {
        int rand_idx = rand() % this->ImageArray.size();
        if (closed_set.find(rand_idx) != closed_set.end())
            continue;
        switch (arrayclass)
        {
            case 0: this->TrainArray.push_back(this->ImageArray[rand_idx]);
            break;
        
            case 1: this->TestArray.push_back(this->ImageArray[rand_idx]);
            break;

            case 2: this->ValidationArray.push_back(this->ImageArray[rand_idx]);
            break;
        }
        closed_set.emplace(rand_idx);
    }
}*/

void brainiac::Imgdata_handler::threeway_split_data () {
    /*std::unordered_set<int> closed_set;
    uint32_t datasize = this->TrainArray.size();
    closed_set.reserve(datasize);
    
    uint16_t train_size = datasize * this->TRAIN_RATIO;
    this->fillRandom(train_size, closed_set, 0);

    uint16_t test_size = datasize * this->TEST_RATIO;
    this->fillRandom(test_size, closed_set, 1);

    uint16_t val_size = datasize * (1.0 - this->TRAIN_RATIO + this->TEST_RATIO);
    this->fillRandom(val_size, closed_set, 2);
    printf("Successfully split data array of size %d into train, test and validation sets of sizes %d, %d and %d respectively...\n", datasize, train_size, test_size, val_size);*/

    uint32_t size = this->ImageArray.size();
    std::vector<uint32_t> indices;
    indices.reserve(size);
    for (int idx = 0; idx < size; idx ++)
        indices.emplace_back(idx);
    
    std::shuffle(indices.begin(), indices.end(), this->entropySource);

    uint32_t count = 0;
    uint32_t train_size = size * this->TRAIN_RATIO;
    uint32_t test_size  = size * this->TEST_RATIO;
    uint32_t val_size   = size - train_size - test_size;

    while (count < train_size)
        this->TrainArray.push_back(this->ImageArray[count++]);
    while (count < train_size + test_size)
        this->TestArray.push_back(this->ImageArray[count++]);
    while (count < train_size + test_size + val_size)
        this->ValidationArray.push_back(this->ImageArray[count++]);
    printf("Successfully split data array of size %d into train, test and validation sets of sizes %d, %d and %d respectively...\n", size, train_size, test_size, val_size);
}

std::vector<brainiac::Imgdata*> brainiac::Imgdata_handler::getTrainSet () {
    return this->TrainArray;
}
std::vector<brainiac::Imgdata*> brainiac::Imgdata_handler::getTestSet () {
    return this->TestArray;
}
std::vector<brainiac::Imgdata*> brainiac::Imgdata_handler::getValSet () {
    return this->ValidationArray;
}
int brainiac::Imgdata_handler::classCount () {
    return this->class_count_map.size();
}

brainiac::Imgdata_handler::~Imgdata_handler () {
    for (int i = 0; i < this->ImageArray.size(); i ++)
        delete this->ImageArray[i];
}

brainiac::LabelledKmeans::LabelledKmeans (int no_of_classes, int no_of_iterations) {
    this->epochs = no_of_iterations;
    this->K = no_of_classes;

    this->entropyEngine.seed(this->entropySource());

    this->centroids.reserve(no_of_classes);

    this->Clusters.resize(no_of_classes);
    this->dominant_label_maps.resize(no_of_classes);
}
/*
double brainiac::LabelledKmeans::ImageMean (brainiac::Imgdata* img) {
    double mean;
    std::vector<uint8_t> features = img->getFeatureVector();
    for (int p_ind = 0; p_ind < img->getFeatureSize(); p_ind ++) {
        mean += features[p_ind];
    }
    return (mean / img->getFeatureSize());
}*/

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
    printf("Entered Training..\n");
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
    printf("Completed Training Kmeans classifier on training images...\n");
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
    printf("Predicted : %hhu :: Expected %hhu...\n", max_label, testimg->getClassLabel());
    return max_label;
}




int main () {
    brainiac::Imgdata_handler ImgDataset("data/MNIST_imgs/train-images.idx3-ubyte", "data/MNIST_labels/train-labels.idx1-ubyte");
    ImgDataset.threeway_split_data();
    auto testset = ImgDataset.getTestSet();
    
    brainiac::LabelledKmeans kmeans_classifier(10, 2);
    kmeans_classifier.Train(ImgDataset.getValSet());
    kmeans_classifier.Predict(testset[0]);
    kmeans_classifier.Predict(testset[1]);
    kmeans_classifier.Predict(testset[2]);

    return 0;
}