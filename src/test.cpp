#include "Imgdata_handler.hpp"
#include "Kmeans.hpp"
#include <chrono>
#include <iostream>

using namespace brainiac;

int main () {
    
    // auto t_now = std::chrono::system_clock::now();
    // srand(std::chrono::system_clock::to_time_t(t_now));

    // auto startclock = std::chrono::steady_clock::now();
    
    Imgdata_handler ImgDataset("../data/MNIST_imgs/train-images.idx3-ubyte", "../data/MNIST_labels/train-labels.idx1-ubyte");
    ImgDataset.threeway_split_data();
    
    /*KNN knn_classifier(5);
    knn_classifier.LoadTraindata(ImgDataset.getTrainSet());
    double performance = knn_classifier.ValidatePerformance(ImgDataset.getValSet());*/
    
    LabelledKmeans kmeans_classifier(10, 20);
    kmeans_classifier.Train(ImgDataset.getTrainSet());

    kmeans_classifier.ValidatePerformance(ImgDataset.getValSet());
    /*kmeans_classifier.Train(ImgDataset.getValSet());
    kmeans_classifier.Predict(testset[0]);
    kmeans_classifier.Predict(testset[1]);
    kmeans_classifier.Predict(testset[2]);
    kmeans_classifier.Predict(testset[3]);
    kmeans_classifier.Predict(testset[4]);
    kmeans_classifier.Predict(testset[5]);
    kmeans_classifier.Predict(testset[6]);
    kmeans_classifier.Predict(testset[7]);
    kmeans_classifier.Predict(testset[8]);
    kmeans_classifier.Predict(testset[9]);*/

    // auto endclock = std::chrono::steady_clock::now();
    // std::cout<<"\n" <<std::chrono::duration_cast<std::chrono::microseconds>(endclock - startclock).count()<< " µs\n";

    return 0;
}