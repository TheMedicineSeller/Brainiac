#include "Csvdata_handler.hpp"

brainiac::Csvdata_handler::Csvdata_handler (const char* filePath) {
    int num_classes = 0, delimiter_pos = 0;
    std::string buffer;
    std::ifstream csvHandle(filePath);

    while (std::getline(csvHandle, buffer)) {

        if (buffer.length() == 0) continue;
        std::string feature;
        brainiac::Csvdata* entry = new Csvdata();
        while ((delimiter_pos = buffer.find(',')) != std::string::npos) {
            feature = buffer.substr(0, delimiter_pos);
            entry->AppendtoFeatures(std::stod(feature));
            buffer.erase(0, delimiter_pos + 1);
        }

        if (this->class_count_map.find(buffer) == this->class_count_map.end())
            this->class_count_map[buffer] = ++num_classes;
        entry->SetLabel(this->class_count_map[buffer]);
        this->data.push_back(entry);
    }
    for (int i = 0; i < this->data.size(); i ++) {
        this->data[i]->EncodeLabel(num_classes);
    }
    csvHandle.close();
}

void brainiac::Csvdata_handler::three_way_split (const double trainpercentage, const double testpercentage) {
    std::vector<int> indices(this->data.size());
    std::generate(indices.begin(), indices.end(), [n = 0]() mutable { return n++; });

    std::shuffle(indices.begin(), indices.end(), this->entropySource);

    int trainsize = trainpercentage * this->data.size();
    int testsize  = testpercentage  * this->data.size();
    int valsize   = this->data.size() - trainsize - testsize,
        count       = 0;
    
    this->train_data.reserve(trainsize);
    this->test_data.reserve(testsize);
    this->validation_data.reserve(valsize);

    while (count < trainsize)
        this->train_data.emplace_back(this->data[indices[count++]]);
    while (count < trainsize + testsize)
        this->test_data.emplace_back(this->data[indices[count++]]);
    while (count < trainsize + testsize + valsize)
        this->validation_data.emplace_back(this->data[indices[count++]]);

    printf("Successfully split data array of size %d into train, test and validation sets of sizes %d, %d and %d respectively...\n", this->data.size(), trainsize, testsize, valsize);
}

std::vector<brainiac::Csvdata*> brainiac::Csvdata_handler::getTestData () {
    return this->test_data;
}

std::vector<brainiac::Csvdata*> brainiac::Csvdata_handler::getValidationData () {
    return this->validation_data;
}

brainiac::Csvdata_handler::~Csvdata_handler () {
    for (int i = 0; i < this->data.size(); i ++)
        delete this->data[i];
}