#include "Imgdata_handler.hpp"

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
