### Run.sh 
```
#!/bin/bash
make
./docSeg gmmGbrTr "/users/jobinkv/threeClassTrainData/newTrain" "/users/jobinkv/threeClassTrainData/newtrGt" tdeepfet.xml
mkdir -p output
./docSeg gmmGrTestF "/users/jobinkv/threeClassTrainData/testImg" "./output" tdeepfet.xml
```
### main.cpp
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
### function.cpp


