#include <vector>
#include <fstream>
#include <string>
#include <unordered_map>
#include <stdio.h>

namespace brainiac {
    class Csvdata {
        std::vector<double> features;
        int label;
        std::vector<int> OneHotEncodedLabel;
        static int num_classes;
        public:
            Csvdata(int no_of_features) {
                this->features.reserve(no_of_features);
            }
            void SetFeatures(const std::vector<double> &_features) {
                this->features = _features;
            }
            void AppendtoFeatures(double parameter) {
                this->features.emplace_back(parameter);
            }
            void SetLabel(int _label) {
                this->label = _label;
            }
            void EncodeLabel(int num_classes) {
                this->OneHotEncodedLabel.assign(num_classes, 0);
                this->OneHotEncodedLabel[this->label] = 1;
            }

            std::vector<double> getFeatures() {
                return this->features;
            }
            int getClassLabel() {
                return this->label;
            }
    };
}

namespace brainiac {
    class Csvdata_handler {
        std::vector<Csvdata*> data;
        std::vector<Csvdata*> train_data;
        std::vector<Csvdata*> test_data;
        std::vector<Csvdata*> validation_data;

        std::unordered_map<std::string, int> class_count_map;
        
        int count_lines(std::ifstream &csvHandle);
        public:
            Csvdata_handler(const char* filePath);
            void three_way_split(const double trainpercentage=0.7, const double testpercentage=0.2);
            std::vector<Csvdata*> getTestSet();
            std::vector<Csvdata*> getValidationSet();
            ~Csvdata_handler();
    };
}

int brainiac::Csvdata_handler::count_lines (std::ifstream &csvHandle) {
    std::string linebuf;
    int linecount = 0;
    while (std::getline(csvHandle, linebuf))
        linecount ++;
    csvHandle.seekg(0, std::ios::beg);
    return linecount;
}

brainiac::Csvdata_handler::Csvdata_handler (const char* filePath) {
    int num_classes = 0, delimiter_pos;
    std::string buffer, feature;
    std::ifstream csvHandle(filePath);
    
    int line_count = this->count_lines(csvHandle);
    this->data.reserve(line_count);
    // Advancing the file pointer by one row
    std::getline(csvHandle, buffer);

    while (std::getline(csvHandle, buffer)) {
        if (buffer.length() == 0) continue;
        brainiac::Csvdata* entry = new Csvdata(line_count);
        while ((delimiter_pos = buffer.find(',')) != std::string::npos) {
            feature = buffer.substr(0, delimiter_pos);
            entry->AppendtoFeatures(std::stod(feature));
            buffer.erase(0, delimiter_pos + 1);
        }
        if (this->class_count_map.find(buffer) == this->class_count_map.end())
            this->class_count_map[buffer] = ++num_classes;
        entry->SetLabel(this->class_count_map[buffer]);
        this->data.emplace_back(entry);
    }
    for (int i = 0; i < this->data.size(); i ++) {
        this->data[i]->EncodeLabel(num_classes);
    }
    csvHandle.close();
}

/*Simpler three way splits without employing any randomness in split. Takes contiguos subarrays of respective sizes for train, test and validation*/
void brainiac::Csvdata_handler::three_way_split (const double trainpercentage, const double testpercentage) {
    int trainsize = trainpercentage * this->data.size(),
        testsize  = testpercentage  * this->data.size(),
        valsize   = this->data.size() - trainsize - testsize,
        itr       = 0;
    
    this->train_data.reserve(trainsize);
    this->test_data.reserve(testsize);
    this->validation_data.reserve(valsize);
    
    for (; itr < trainsize; itr ++) {
        this->train_data.emplace_back(this->data[itr]);
    }
    for (; itr < testsize; itr ++) {
        this->test_data.emplace_back(this->data[itr]);
    }
    for (; itr < valsize; itr ++) {
        this->validation_data.emplace_back(this->data[itr]);
    }
}

std::vector<brainiac::Csvdata*> brainiac::Csvdata_handler::getTestSet () {
    return this->test_data;
}

std::vector<brainiac::Csvdata*> brainiac::Csvdata_handler::getValidationSet () {
    return this->validation_data;
}

brainiac::Csvdata_handler::~Csvdata_handler () {
    for (int i = 0; i < this->data.size(); i ++)
        delete this->data[i];
}

int main () {
    brainiac::Csvdata_handler csvDataset("../data/CSV/Iris.csv");
    csvDataset.three_way_split();
    std::vector<brainiac::Csvdata*> testset = csvDataset.getTestSet();

    printf("size = %d", testset.size());
    for (int i = 0; i < testset.size(); i ++)
    {
        printf("four\n");
        printf("%d \n", testset[i]->getClassLabel());
    }

    return 0;
}