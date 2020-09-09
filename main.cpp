#include <timer.h>
#include <betterthanmnist.h>
#include <neuralnetwork.h>

using namespace std;

char intToChar(unsigned i){
    return char(i + 65);
}

float GetDataAccuracy (BetterThanMnist& data, NeuralNetwork& neuralNetwork, unsigned numTests)
{
    unsigned correctItems = 0;
    for (unsigned i = 0, c = numTests; i < c; ++i)
    {
        unsigned label;
        vector<float> pixels = data.GetImage(label);
        unsigned detectedLabel = neuralNetwork.ForwardPass(pixels);

        if (detectedLabel == label)
            ++correctItems;
    }
    return float(correctItems) / float(numTests);
}

void ShowImage (vector<float>& pixels)
{
    string render="";
    for (unsigned i = 0; i< 784; ++i){
        if (i % 28 == 0)
            render += '\n';
        if (pixels.at(i) == 1.0f)
            render += '#';
        else
            render += '.';
    }
    cout<<render<<endl;
}

void writeFloatsInFile(fstream& fout, const float* data, const unsigned arr_size){
    for(unsigned i = 0;i < arr_size;++i)
        fout<< data[i]<< ' ';
    fout<<endl;
}

void testNetwork(NeuralNetwork& neuralNetwork, BetterThanMnist& picture, unsigned testsNum){
    float accuracyTest = GetDataAccuracy(picture, neuralNetwork, testsNum);
    cout<<"Test network Accuracy: "<< 100.0f * accuracyTest<<"%"<<endl;
}

void saveResults(NeuralNetwork& neuralNetwork){
    fstream fout;
    fout.open(c_init_filename);
    const float* data = neuralNetwork.GetHiddenLayerBiases();
    writeFloatsInFile(fout, data, c_numHiddenNeurons);

    data = neuralNetwork.GetOutputLayerBiases();
    writeFloatsInFile(fout, data, c_numOutputNeurons);

    data = neuralNetwork.GetHiddenLayerWeights();
    writeFloatsInFile(fout, data, c_numInputNeurons*c_numHiddenNeurons);

    data = neuralNetwork.GetOutputLayerWeights();
    writeFloatsInFile(fout, data, c_numHiddenNeurons*c_numOutputNeurons);

    cout<<"[+] Results saved."<<endl;
}

void do_honset_test(NeuralNetwork& neuralNetwork, BetterThanMnist& picture, string filename){
    vector<float> pic = picture.GetTestImage(filename);
    unsigned detectedNum = neuralNetwork.ForwardPass(pic);
    ShowImage(pic);
    cout<<"Detected number: "<<intToChar(detectedNum);
}

void trainingNetwork(NeuralNetwork& neuralNetwork, BetterThanMnist& picture){
    Clock timer("Training Time:  ");

    cout<<endl<<"[+] Training started--------------------------------"<<endl;
    for (unsigned generation = 0; generation < c_trainingGenerations; ++generation){
        neuralNetwork.Train(picture);

        cout<<"Training generation "<<generation + 1<<" / "<< c_trainingGenerations<<' ';
        float accuracy = GetDataAccuracy(picture, neuralNetwork, 20);
        cout<<"Test accuracy: "<<100.0f*accuracy<<'%'<<endl;
    }

    cout<<picture.NumImages()<<" tests passed."<<endl<<endl;
    timer.get_info();
    cout<<"[+] Training finished-------------------------------"<<endl<<"[+] Now testing."<<endl<<endl;

    float accuracyTest = GetDataAccuracy(picture, neuralNetwork, 40);
    cout<<"Final Accuracy Test: "<< 100.0f * accuracyTest<<"%"<<endl;

    cout<<endl<<"Do you want to save training results? (Y/N)"<<endl;
    string cmd; cin>>cmd;
    if (cmd == "Y"){
        saveResults(neuralNetwork);
    }
    else{
        cout<<"Results aren\'t saved"<<endl;
    }

}


void automaticTraining(NeuralNetwork& oldNet, BetterThanMnist& picture){

    picture.reopen();
    float currentAccuracy = GetDataAccuracy(picture, oldNet, 10000);
    cout<<"Old accuracy: "<<currentAccuracy*100<<"%"<<endl<<endl;
    picture.reopen();

    unsigned short bad_try_count = 0;

    Clock timer("Training Time:  ");

    bool exit = true;
    cout<<endl<<"[+] Automatic training started"<<endl;
    do{
        cout<<"------------------------------"<<endl;
        srand(time(0));
        unsigned new_batch_size = rand()%10 + 2;
        float new_learnig_rate = random_device()()/(random_device().max()/3.2f) +.2f;
        NeuralNetwork net(new_batch_size, new_learnig_rate);
        net.initialize();
        cout<<"New bacth: "<<new_batch_size<<' '<<"New rate: "<<new_learnig_rate<<endl;

        for (unsigned generation = 0; generation < c_trainingGenerations; ++generation)
            net.Train(picture);
        cout<<"[+] tests passed."<<endl;

        picture.reopen();
        float newAccuracy = GetDataAccuracy(picture, net, 10000);
        cout<<"New accuracy: "<<newAccuracy*100<<"%"<<endl;
        picture.reopen();

        if (newAccuracy > currentAccuracy){
            saveResults(net);
            currentAccuracy = newAccuracy;
            bad_try_count = 0;
        }else {
            ++bad_try_count;
        }

        if (bad_try_count == 10)
            exit = false;

    }while(exit);

    timer.get_info();


}

int main (){
    NeuralNetwork neuralNetwork;
    BetterThanMnist picture;


    bool exit = true;
    do {
        cout<<endl<<"Enter 1 to test network."<<endl;
        cout<<"Enter 2 to initialize network."<<endl;
        cout<<"Enter 3 to train network."<<endl;
        cout<<"Enter 4 to do ONE recognition from your file."<<endl;
        cout<<"Enter 5 to reopen test file."<<endl;
        cout<<"Enter 6 to save weights."<<endl;
        cout<<"Enter 7 to automatic training."<<endl;
        cout<<"Enter 0 to exit"<<endl<<endl;

        char cmd; cin>>cmd;

        if (isdigit(cmd)){
            int choice = int(cmd) - 48;
            switch (choice) {
                case 0:
                    exit = false;
                    break;
                case 1:
                    cout<<endl<<"How many tests do you want?"<<endl;
                    unsigned b; cin>>b;
                    testNetwork(neuralNetwork, picture, b);
                    break;
                case 2:
                    neuralNetwork.initialize();
                    break;
                case 3:
                    trainingNetwork(neuralNetwork, picture);
                    break;
                case 5:
                    picture.reopen();
                    break;
                case 6:
                    saveResults(neuralNetwork);
                    break;
                case 7:
                    automaticTraining(neuralNetwork, picture);
                    break;
                case 4:
                    string file;cout<<"Enter filename: ";cin>>file;
                    do_honset_test(neuralNetwork, picture, file);
                    break;

            }
        }else {
            exit = false;
            cout<<"Wrong input!";
        }
    }while(exit);

    return 0;
}
