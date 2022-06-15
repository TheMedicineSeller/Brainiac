// #include "Imgdata_handler.hpp"
// #include "KNN.hpp"
// #include "Kmeans.hpp"
#include "Csvdata_handler.hpp"
// // #include <chrono>
// #include <iostream>

using namespace brainiac;

int main () {
    
    // auto t_now = std::chrono::system_clock::now();
    // srand(std::chrono::system_clock::to_time_t(t_now));

    // auto startclock = std::chrono::steady_clock::now();
    /* --------------------KNN--------------------
    Imgdata_handler ImgDataset("../data/MNIST_imgs/train-images.idx3-ubyte", "../data/MNIST_labels/train-labels.idx1-ubyte");
    ImgDataset.threeway_split_data();
    
    KNN knn_classifier(10);
    knn_classifier.LoadTraindata(ImgDataset.getTrainSet());
    double performance = knn_classifier.ValidatePerformance(ImgDataset.getValSet());*/
    
    /* --------------------Kmeans--------------------
    LabelledKmeans kmeans_classifier(10, 20);
    kmeans_classifier.Train(ImgDataset.getTrainSet());
    kmeans_classifier.ValidatePerformance(ImgDataset.getValSet());*/

    Csvdata_handler csvDataset("../data/CSV/Iris.csv");
    csvDataset.three_way_split();
    auto valset = csvDataset.getValidationData();
    printf("size = %d", valset.size());
    for (int i = 0; i < valset.size(); i ++)
    {
        printf("%d \n", valset[i]->getClassLabel());
    }
    
    // auto endclock = std::chrono::steady_clock::now();
    // std::cout<<"\n" <<std::chrono::duration_cast<std::chrono::microseconds>(endclock - startclock).count()<< " µs\n";

    return 0;
}