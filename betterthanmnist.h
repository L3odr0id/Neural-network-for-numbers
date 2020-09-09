#ifndef BETTERTHANMNIST_H
#define BETTERTHANMNIST_H

#include <vector>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;

const string c_dataFileName = "learningData.txt";


class BetterThanMnist{
public:
    BetterThanMnist(){
        inputFile.open(c_dataFileName);
        cout<<"[+] File "<<c_dataFileName<<" opened."<<endl;

    }

    unsigned NumImages () const { return imageCount; }

    bool LoadNextPicture(){
        pixels.clear();
        string check;
        inputFile >> check;
        pixels.resize(784);
        for (unsigned i = 0; i < 784;++i){
            float a; inputFile >> a;
            pixels.at(i) = a;
        }
        inputFile >> check;
        if (check == "out:")
            inputFile >> label;
        return true;
    }

    vector<float> GetTestImage(const string filename){
        fstream fin;
        fin.open(filename);
        string check;
        fin >> check;
        pixels.resize(784);
        for (unsigned i = 0; i < 784;++i){
            float a; fin >> a;
            pixels.at(i) = a;
        }
        fin.close();
        return pixels;
    }

    void reopen(){
        inputFile.close();
        inputFile.open(c_dataFileName);
        cout<<"[+] File "<<c_dataFileName<<" reopened."<<endl;
    }

    vector<float> GetImage (unsigned& label_out, bool newPic = true)
    {
        if (newPic)
            this->LoadNextPicture();
        label_out = label;
        ++imageCount;
        return pixels;
    }
private:
    unsigned imageCount = 0;
    unsigned label = 0;
    vector<float> pixels;
    fstream inputFile;
};

#endif // BETTERTHANMNIST_H
