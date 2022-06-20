#pragma once

#include "Imgdata.hpp"
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

namespace brainiac
{
    namespace nn
    {
        inline double heaviside_activation (double x) {
            return (x > 0) ? 1. : 0.;
        }
        inline double relu_activation (double x) {
            return (x >= 0) ? x : 0.;
        }
        inline double leakyrelu_activation (double x) {
            return (x >= 0) ? x : 0.02 * x;
        }
        inline double sigmoid_activation (double x) {
            return (1 / (1 + std::exp(-x)));
        }

        struct Neuron {
            double value;
            double activated_value;
            double error;
            double gradient;

            double sigmoid_derivative (int x) {
                double sig = sigmoid_activation(x);
                return sig * (1 - sig);
            }
            void setGradient () {
                this->gradient = this->error * this->sigmoid_derivative(this->activated_value);
            }
        };

        typedef std::vector<std::vector<double>> matrix_t;
        typedef std::vector<Neuron> layer;

        std::vector<double> VectorTransform(const matrix_t &matrix, const layer &LayerVector);


        class TrainingNetwork {
            static double bias;
            double learningrate;
            std::vector<layer> network;
            std::vector<matrix_t> weightTransforms;
            
            void FeedForward(const std::vector<double> &inputs);
            void BackPropagate(const std::vector<double> &Expected_activations);
            
            public:
                TrainingNetwork(const std::vector<int> &topology, double lr=0.05);
                void Train(const std::vector<Imgdata*> &Train_images, int epochs);
                int classifyImage(Imgdata* img);
                void ValidatePerformance(const std::vector<Imgdata*> &Validation_images);
         };
    }
}