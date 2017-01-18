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

``` c++
void TrainGmmGbrModel(string org_folder,string gt_folder,string model)// dealing wi th afolder
{
                vector<ThreeMatstr> CF;
                vector<string> files = vector<string>();
                getdir(org_folder,files);
                vector<ThreeMatstr>  feturelist;

                ::google::InitGoogleLogging("./docSeg");
                string model_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/deploy.prototxt";
                string trained_file = "/users/jobinkv/installs/caffe_cpp/googleNet/bvlc_reference_caffenet.caffemodel";
                string mean_file    = "/users/jobinkv/installs/caffe_cpp/googleNet/imagenet_mean.binaryproto";
                string label_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/synset_words.txt";
                Classifier deepFt(model_file, trained_file, mean_file, label_file);

                for (unsigned int i = 0;i < files.size();i++)
                {

                        string gt_path =  gt_folder + string("/") + files[i] ;
                        string org_path = org_folder + string("/") + files[i] ;

                        Mat image, gt_img;
                        image = imread(org_path, CV_LOAD_IMAGE_COLOR);   // Reading original image
                        gt_img = imread(gt_path, CV_LOAD_IMAGE_COLOR);   // Reading gt image
                        if( gt_img.cols != image.cols or gt_img.rows != image.rows )
                        {
                                cout <<"ERROr : Image dimentios of the given images "<<files[i]<<" are not matching" << endl;
                                continue;
                        }

                        cout<<"Now learning "<<files[i]<<endl;

                        ThreeMatstr train_feture = crtTrainFetGabur(image,gt_img,deepFt);
                        feturelist.push_back(train_feture);
                        if(i>=3) break;
                }

                ThreeMatstr clean_feture = list2mat1(feturelist);

                Mat textFeture = L2Normalization(clean_feture.text.t());
                Mat figue =L2Normalization(clean_feture.figure.t());
                Mat backGnd = L2Normalization(clean_feture.background.t());
                // gmm model creation
                // gmm statrs/=====================================---------------------
                const int cov_mat_type = cv::EM::COV_MAT_SPHERICAL;//COV_MAT_GENERIC;//
                cv::TermCriteria term(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 15000, 1e-3);
                cv::EM gmm_text(NofG, cov_mat_type, term);
                cv::EM gmm_grapics(NofG, cov_mat_type, term);
                cv::EM gmm_backgnd(NofG, cov_mat_type, term);

                // Training of GMM              
                cout << "Training GMM... " << flush;
                bool status = gmm_text.train(textFeture);
                status = gmm_grapics.train(figue);
                status = gmm_backgnd.train(backGnd);
                cout << "Done! " << endl;
                // writing attempt
                //string model ("model.xml");
                string modTxt ("text_");
                string fulname = modTxt+model;
                gmmSave(gmm_text,fulname);
                //==========================
                string modGra ("gra_");
                fulname = modGra+model;
                gmmSave(gmm_grapics,fulname);
                //==========================
                string modBac ("bac_");
                fulname = modBac+model;
                gmmSave(gmm_backgnd,fulname);

}

Mat gmmGrtest(Mat image,string model,Classifier deep)
{
        Mat gray;
        cvtColor(image,gray,CV_RGB2GRAY);
        // declairing feature extraction class
        fetExtrct meth1;
        // setting input image
        meth1.setInpImage(image);

        ///gaburFet
        if (feturE==1)
                meth1.gaburFet ();

        // lbp fet
        if (feturE==2)
                meth1.lbpFet();

        // cc features
        if (feturE==3)
                meth1.ccFet ();


        //::google::InitGoogleLogging("./docSeg");
        //string model_file   = "deploy.prototxt";
        //string trained_file = "bvlc_reference_caffenet.caffemodel";
        //string mean_file    = "imagenet_mean.binaryproto";
        //string label_file   = "synset_words.txt";
        //Classifier classifier(model_file, trained_file, mean_file, label_file);

        // take the list of patches
        Mat listofpatch = meth1.listOfpatch();

        Mat outImage(image.rows,image.cols, CV_8UC3, Scalar(0,0,0));


        EM gmmText;// = allModel.text;
        EM gmmGraph;// = allModel.figure;
        EM gmmBacK;// =  allModel.background;
        // reading attempt
        //string model ("model.xml");
        //==============================
        string modTxt ("text_");
        string fulname = modTxt+model;
        gmmText = readModel(fulname);
        //=============================
        string modGra ("gra_");
        fulname = modGra+model;
        gmmGraph = readModel(fulname);
        //===========================
        string modbac ("bac_");
        fulname = modbac+model;
        gmmBacK = readModel(fulname);

        Mat hist;

        for (int i=0;i<listofpatch.rows;i++)//
        {
                Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),p_size,p_size);
                if (feturE==1)
                hist = meth1.features(ross);// gabour
                else if (feturE==2)
                hist = meth1.lbpftr(ross); //lbp features
                else if (feturE==3)
                hist = meth1.ccftrXtr(ross); //cc ftr
                else if (feturE==4){
                Mat patch = gray(ross).clone();
                hist = deep.Classify(patch);
                }
                double txt = gmmRealPred( gmmText,L2Normalization( hist));
                double graP = gmmRealPred( gmmGraph, L2Normalization(hist));
                double bac = gmmRealPred( gmmBacK, L2Normalization(hist));
                rectangle(outImage, ross, Scalar((int)(txt),(int)(graP),(int)(bac)), -1, 8, 0 );
        }
        // clear model
        meth1.clrAll();
        Mat enerfyMin;
//////////////////////// Alpha expansion //////////////////////
        int num_labels = 3;
        int lambada=.45*255;
        Mat downSmp;
        resize(outImage, downSmp, Size(),(double)1/p_size, (double)1/p_size, INTER_NEAREST);
                // smoothness and data costs are set up one by one, individually
        //namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
        //imshow( "Display window", outImage ); 
        //waitKey(0);   
        //clock_t tStart = clock();
        //enerfyMin = rectPrior1(outImage);     
        //printf("Time taken for rectPrior1: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        //enerfyMin = segmentRectPrior(outImage);       
        enerfyMin = GridGraph_Individually(num_labels,downSmp,lambada);
        // resize as the size of the original image
        //namedWindow( "Display window1", WINDOW_NORMAL );// Create a window for display.
        //imshow( "Display window1", enerfyMin ); 
        //waitKey(0); 
        resize(enerfyMin, enerfyMin, image.size(), INTER_NEAREST);
        Mat enerfyMinSplit[3];
        split(enerfyMin, enerfyMinSplit);
        for (int i=0;i<3;i++)
                threshold(enerfyMinSplit[i],enerfyMinSplit[i],125,255,THRESH_BINARY);
        merge(enerfyMinSplit,3,enerfyMin);
        return enerfyMin;
}



```
