#include "Imgdata_handler.hpp"
#include "KNN.hpp"
#include <chrono>
#include <iostream>

using namespace brainiac;

int main () {
    
    // auto t_now = std::chrono::system_clock::now();
    // srand(std::chrono::system_clock::to_time_t(t_now));

    Imgdata_handler ImgDataset("../data/MNIST_imgs/train-images.idx3-ubyte", "../data/MNIST_labels/train-labels.idx1-ubyte");
    
    ImgDataset.threeway_split_data();
    
    KNN knn_classifier(5);
    knn_classifier.LoadTraindata(ImgDataset.getTrainSet());
    // std::vector<Imgdata*> testimg = ImgDataset.getTestSet();
    // std::cout<< "Class predicted for test image 1: "<< static_cast<int>(knn_classifier.Predict(testimg[0]))<< "\n";
    // std::cout<< "Class predicted for test image 2: "<< static_cast<int>(knn_classifier.Predict(testimg[1]))<< "\n";
    // std::cout<< "Class predicted for test image 3: "<< static_cast<int>(knn_classifier.Predict(testimg[2]))<< "\n";
    auto startclock = std::chrono::steady_clock::now();
    double performance = knn_classifier.ValidatePerformance(ImgDataset.getValSet());
    auto endclock = std::chrono::steady_clock::now();
    
    std::cout<< "Performace of the classifier over the given validation set: "<< performance<< "\n";
    std::cout<<"\n" <<std::chrono::duration_cast<std::chrono::microseconds>(endclock - startclock).count()<< " µs\n";

    return 0;
}