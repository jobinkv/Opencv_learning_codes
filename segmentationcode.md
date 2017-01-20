# Segmentation code

## Table of Contents
* **[Shell script](#shell-script)** 
* **[GMM based functions in main file](#gmm-based-functions-in-main-file)**
* **[SVM based function in main file](#svm-based-function-in-main-file)**
* **[GMM training](#gmm-training)**
* **[GMM testing](#gmm-testing)**
* **[SVM training](#svm-training)**
* **[SVM testing](#svm-testing)**
* **[Common feature extraction function for training](#common-feature-extraction-function-for-training)**



## Shell script
#### GMM based training
``` bash
#!/bin/bash
make
./docSeg gmmGbrTr "/users/jobinkv/threeClassTrainData/newTrain" "/users/jobinkv/threeClassTrainData/newtrGt" tdeepfet.xml
mkdir -p output
./docSeg gmmGrTestF "/users/jobinkv/threeClassTrainData/testImg" "./output" tdeepfet.xml
```
#### SVM based training
``` bash
./docSeg train "/users/jobinkv/threeClassTrainData/newTrain" "/users/jobinkv/threeClassTrainData/newtrGt" tdeepfet.xml
mkdir -p output
./docSeg testF "/users/jobinkv/threeClassTrainData/testImg" "./output" tdeepfet.xml

```
[Return to main menu](#table-of-contents)

### GMM based functions in main file 

``` c++
        else if(mode =="gmmGbrTr")// main train funcrion ------------------------------================================
        {
                cout<<"training started"<<endl;
                string org_folder = string(argv[2]);
                string gt_folder = string(argv[3]);
                string model_name =  string(argv[4]);
                // it will create a classifier.xml file
                TrainGmmGbrModel(org_folder,gt_folder, model_name);
        }

        else if (mode =="gmmGrTestF")// gabour featute testing ==================================================
        {
                string model =  argv[4];
                cout<<"tesing started"<<endl;
                // reading all files in the folder
                ::google::InitGoogleLogging("./docSeg");
                string model_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/deploy.prototxt";
                string trained_file = "/users/jobinkv/installs/caffe_cpp/googleNet/bvlc_reference_caffenet.caffemodel";
                string mean_file    = "/users/jobinkv/installs/caffe_cpp/googleNet/imagenet_mean.binaryproto";
                string label_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/synset_words.txt";
                Classifier deepFt(model_file, trained_file, mean_file, label_file);
                //------------------------------------
                vector<string> files = vector<string>();
                getdir(argv[2],files);
                for (unsigned int i = 0;i < files.size();i++)
                {

                        string inputPath =  argv[2] + string("/") + files[i] ;
                        string outputPath =  argv[3] + string("/") + files[i] ;
                        cout<<"Now testing "<<files[i]<<endl;
                        Mat image = imread(inputPath, CV_LOAD_IMAGE_COLOR);
                        clock_t tStart = clock();
                        Mat layout = gmmGrtest(image, model,deepFt);//char *model_readed
                        printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
                        imwrite(outputPath,layout);

                }

        }
```
[Return to main menu](#table-of-contents)

### SVM based function in main file 
``` c++
        else if(mode =="train")
        {
                cout<<"training  started"<<endl;
                string org_folder = string(argv[2]);
                string gt_folder = string(argv[3]);
                char* model_name =  argv[4];
                // it will create a classifier.xml file
                TrainTheModel(org_folder,gt_folder, model_name);
        }
        if (mode =="testF")
        {
                char* model_readed =  argv[4];
                cout<<"tesing started"<<endl;
                // reading all files in the folder
                vector<string> files = vector<string>();
                getdir(argv[2],files);

                string model_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/deploy.prototxt";
                string trained_file = "/users/jobinkv/installs/caffe_cpp/googleNet/bvlc_reference_caffenet.caffemodel";
                string mean_file    = "/users/jobinkv/installs/caffe_cpp/googleNet/imagenet_mean.binaryproto";
                string label_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/synset_words.txt";
                Classifier deepFt(model_file, trained_file, mean_file, label_file);

                for (unsigned int i = 0;i < files.size();i++)
                {
                        string inputPath =  argv[2] + string("/") + files[i] ;
                        string outputPath =  argv[3] + string("/") + files[i] ;
                        cout<<"Now running "<<files[i]<<endl;
                        Mat image = imread(inputPath, CV_LOAD_IMAGE_COLOR);
                        Mat layout = docLayotSeg(image, model_readed,deepFt);//char *model_readed
                        imwrite(outputPath,layout);

                }

        }


```
[Return to main menu](#table-of-contents)
### GMM training 
code [TrainGmmGbrModel](functionMain.cpp#L2129-L2198)

### GMM testing
code [gmmGrtest](functionMain.cpp#L2320-L2421)

### SVM training
code [TrainTheModel](functionMain.cpp#L1296-L1433)


### SVM testing
code [docLayotSeg](functionMain.cpp#L1525-L1546)
code [crtTestFet](functionMain.cpp#L1436-L1522)

### Common feature extraction function for training
code [crtTrainFetGabur](functionMain.cpp#L2202-L2294)


[Return to main menu](#table-of-contents)
