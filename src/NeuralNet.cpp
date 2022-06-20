#include "NeuralNet.hpp"

std::vector<double> brainiac::nn::VectorTransform (const matrix_t &matrix, const layer &LayerVector) {
    std::vector<double> OutVector(matrix.size());
    for (int i = 0; i < matrix.size(); i ++) {
      double prod = 0.;
      for (int j = 0; j < LayerVector.size(); j ++) {
        prod += LayerVector[j].activated_value * matrix[i][j];
      }
      OutVector[i] = prod;
    }
    return OutVector;
}

double brainiac::nn::TrainingNetwork::bias = 1.;

brainiac::nn::TrainingNetwork::TrainingNetwork (const std::vector<int> &topology, double lr) : learningrate(lr) {

    srand(time(NULL));

    for (int i = 0 ; i < topology.size() - 1; i ++) {
        /*+1 additional Neuron and corresponding weight for the bias*/
        this->network.push_back(layer(topology[i] + 1));
        // only add bias to col
        this->weightTransforms.push_back(matrix_t(topology[i+1], std::vector<double>(topology[i] + 1)));
    }
    this->network.push_back(layer(topology.back()));

    for (auto &_matrix : this->weightTransforms) {
        for (int i = 0; i < _matrix.size(); i ++)
            for (int j = 0; j < _matrix[j].size(); j ++)
                _matrix[i][j] = rand() / RAND_MAX;
    }
}

void brainiac::nn::TrainingNetwork::FeedForward (const std::vector<double> &inputs) {
    /*Appending additionaly to inputs a constant bias*/

    for (int i = 0; i < this->network[0].size(); i ++) {
        this->network[0][i].value = inputs[i];
        this->network[0][i].activated_value = inputs[i];
    }
    this->network[0].back().value = this->bias;
    this->network[0].back().activated_value = this->bias;

    for (int itr = 1; itr < this->network.size(); itr ++) {
        std::vector<double> outvect = brainiac::nn::VectorTransform(this->weightTransforms[itr - 1], this->network[itr - 1]);
        if (itr != this->network.size() - 1)
            outvect.push_back(this->bias);
        
        for (int i = 0; i < this->network[itr].size(); i ++) {
            this->network[itr][i].value = outvect[i];
            this->network[itr][i].activated_value = sigmoid_activation(outvect[i]);
        }
    }
}

void brainiac::nn::TrainingNetwork::BackPropagate (const std::vector<double> &Expected_activations) {
    for (int i = 0; i < this->network.back().size(); i ++) {
        this->network.back()[i].error = Expected_activations[i] - this->network.back()[i].activated_value;
        this->network.back()[i].setGradient();
    }
    for (int i = this->network.size() - 2; i >= 0; i -- ) {

        for (int j = 0; j < this->network[i].size(); j ++) {
            /*TODO: COLUMN ACCESS OF MATRIX CAN BE PREVENTED TO INCREASE SPEED*/
            double contribution_to_error = 0.;
            for (int k = 0; k < weightTransforms[i].size(); k ++) {
                contribution_to_error += this->network[i+1][k].gradient * weightTransforms[i][k][j] ;
            }
            this->network[i][j].error = contribution_to_error;
            this->network[i][j].setGradient();
        }
    }
    for (int i = 0; i < weightTransforms.size(); i ++) {

        for (int j = 0; j < weightTransforms[i].size(); j ++) {

            for (int k = 0; k < weightTransforms[i][j].size(); k ++)
                weightTransforms[i][j][k] += this->learningrate * network[i][k].activated_value * network[i+1][j].gradient ;
        }
    }
}

void brainiac::nn::TrainingNetwork::Train (const std::vector<Imgdata*> &Train_images, int epochs) {
    int num_classes = this->network.back().size();
    for (int i = 1 ; i <= epochs; i ++) {
        double cumulative_error = 0.;
        for (Imgdata* img : Train_images) {

            this->FeedForward(img->getFeatureVectorDouble());
            
            /*TODO: implement the OHE class getter inside ingdata*/
            
            std::vector<double> expected(num_classes, 0.);
            expected[static_cast<int>(img->getClassLabel())] = 1.;
                        
            this->BackPropagate(expected);

            for (int j = 0; j < num_classes; j ++) {
                cumulative_error += (expected[j] - network.back()[j].activated_value) * (expected[j] - network.back()[j].activated_value);
            }
        }
        printf("Epoch %d completed with a Mean squared error of %lf\n", i, cumulative_error / num_classes);
    }
}

int brainiac::nn::TrainingNetwork::classifyImage (Imgdata* img) {
    this->FeedForward(img->getFeatureVectorDouble());
    int maxActivatedidx = 0;
    for (int i = 1; i < this->network.back().size(); i ++) {
      if (this->network.back()[i].value > this->network.back()[maxActivatedidx].value)
        maxActivatedidx = i;
    }
    return maxActivatedidx;
}

void brainiac::nn::TrainingNetwork::ValidatePerformance (const std::vector<Imgdata*> &Validataion_images) {
    int hits = 0, count = 0;
    for (Imgdata* img : Validataion_images) {
        if (this->classifyImage(img) == static_cast<int>(img->getClassLabel()))
            hits ++;
        printf("Current performance of neural network: %lf\n", (double)hits / ++count);
    }
}