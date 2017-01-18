# Segmentation code

## Table of Contents
* **[Bash scripts](#Run.sh )**
* **[GMM based training (main)](#main.cpp)**
* **[GMM based training (function)](#functionmain.cpp (GMM Train and test))**
















## Run.sh 
#### SVM based training
~~~~ bash
#!/bin/bash
make
./docSeg gmmGbrTr "/users/jobinkv/threeClassTrainData/newTrain" "/users/jobinkv/threeClassTrainData/newtrGt" tdeepfet.xml
mkdir -p output
./docSeg gmmGrTestF "/users/jobinkv/threeClassTrainData/testImg" "./output" tdeepfet.xml
~~~~
#### SVM based training
~~~~ bash
./docSeg train "/users/jobinkv/threeClassTrainData/newTrain" "/users/jobinkv/threeClassTrainData/newtrGt" tdeepfet.xml
mkdir -p output
./docSeg testF "/users/jobinkv/threeClassTrainData/testImg" "./output" tdeepfet.xml

~~~~
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
### functionmain.cpp (GMM Train and test)

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
### SVM in functionmain.cpp
#### Training
``` c++
void TrainTheModel(string org_folder,string gt_folder, char *model_name)
{
                ThreeMatstr finalFet;
                Mat initial(256,1,CV_32F,Scalar(0));
        //cout<<"text size = "<<initial.size()<<endl;
                finalFet.background = initial.clone();
                finalFet.figure         = initial.clone();
                finalFet.text           = initial.clone();

                vector<string> files = vector<string>();
                getdir(org_folder,files);
//      final feture list
                Mat final_lst;
                //cout<<"testt= "<<files.size()<<endl;
                for (unsigned int i = 0;i < files.size();i++)
                {
                        // if (i==1)// looop for sample run
                        //      break;
                        string gt_path =  gt_folder + string("/") + files[i] ;
                        string org_path = org_folder + string("/") + files[i] ;

                        Mat image, gt_img;
                        image = imread(org_path, CV_LOAD_IMAGE_COLOR);   // Reading original image
                        gt_img = imread(gt_path, CV_LOAD_IMAGE_COLOR);   // Reading gt image
                        //--------------------------------------------------------------------
                        resize(gt_img, gt_img, image.size(), INTER_NEAREST);
                        Mat enerfyMinSplit[3];
                        split(gt_img, enerfyMinSplit);
                        for (int j=0;j<3;j++)
                                threshold(enerfyMinSplit[j],enerfyMinSplit[j],125,255,THRESH_BINARY);
                        merge(enerfyMinSplit,3,gt_img);
                        //imwrite(gt_path,gt_img);
                        //continue;

                        //--------------------------------------------------------------------
                        if( gt_img.cols != image.cols or gt_img.rows != image.rows )
                        {
                                cout <<"ERROr : Image dimentios of the given images "<<files[i]<<" are not matching" << endl;
                                continue;
                        }
                        cout<<"Now running "<<files[i]<<endl;
                        Rect imageprp(p_size,0,image.cols,image.rows);
                        Mat locbp = makeLbpImg(image);
                        Mat listofpatch = patchpos(imageprp);
                        ThreeMatstr train_feture = crtTrainFet(listofpatch,locbp,gt_img);

                // concatinating the out put features
                        hconcat(finalFet.text,train_feture.text,finalFet.text);
                        hconcat(finalFet.figure,train_feture.figure,finalFet.figure);
                        hconcat(finalFet.background,train_feture.background,finalFet.background);
                }
                // ThreeMatstr clean_feture = cleanFet(finalFet);
                ThreeMatstr clean_feture = finalFet;
                int maxx = maximum(clean_feture.text.cols,clean_feture.figure.cols,clean_feture.background.cols);
                cout<<"maximum val= "<<maxx<<endl;
                cout<<"text size= "<<clean_feture.text.cols<<endl;
                cout<<"figure size= "<<clean_feture.figure.cols<<endl;
                cout<<"background size= "<<clean_feture.background.cols<<endl;
                Mat trainData;
                hconcat(clean_feture.text,clean_feture.figure,trainData);
                hconcat(trainData,clean_feture.background,trainData);
        // making of labels
                Mat labels;
                Mat lab_text(clean_feture.text.cols,1,CV_32F,Scalar(0));
                Mat lab_figure(clean_feture.figure.cols,1,CV_32F,Scalar(1));
                Mat lab_background(clean_feture.background.cols,1,CV_32F,Scalar(2));
                vconcat(lab_text,lab_figure,labels);
                vconcat(labels,lab_background,labels);
                trainData = trainData.t();
                Mat R = Mat(1, 3, CV_32FC1);
                R.at<float>(0,0)=1;
                R.at<float>(0,1)=1;
                R.at<float>(0,2)=1;
                cout<< "R vector"<<R<<endl;
                CvMat weights = R;
                CvSVMParams  param = CvSVMParams();
                param.svm_type = CvSVM::C_SVC;
                param.kernel_type = CvSVM::RBF;
                param.degree = 0; // for  poly
                param.gamma = 20; // for poly/rbf/sigmoid
                param.coef0 = 0; // for  poly/sigmoid
                param.C = 7; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR  and  CV_SVM_NU_SVR
                param.nu = 0.0; // for  CV_SVM_NU_SVC, CV_SVM_ONE_CLASS , and  CV_SVM_NU_SVR
                param.p = 0.0; // for CV_SVM_EPS_SVR
                param.class_weights = &weights;//[(.6, 0.3,0.1);//NULL;//for CV_SVM_C_SVC
                param.term_crit.type = CV_TERMCRIT_ITER;         //| CV_TERMCRIT_EPS;
                param.term_crit.max_iter = 1000;
                param.term_crit.epsilon = 1e-6;
                CvParamGrid gammaGrid=CvSVM::get_default_grid(CvSVM::GAMMA);
                CvParamGrid pGrid;
                CvParamGrid nuGrid,degreeGrid;
                //gammaGrid.step=0;
                cout << "Starting training process" << endl;
                cout<<"trainData= "<<trainData.rows<<endl;
                CvSVM svm;
                svm.train_auto(trainData, labels, Mat(), Mat(),param, 4,
                        CvSVM::get_default_grid(CvSVM::C), gammaGrid, pGrid, nuGrid, CvSVM::get_default_grid(CvSVM::COEF), degreeGrid, true);
                svm.save(model_name);

                cout << "Finished training process" << endl;
}

```
### svm test in functionmain.cpp
``` c++
Mat docLayotSeg(Mat image, char *model_readed)
{

        Mat enerfyMin;
        Mat outImage = crtTestFet(image, model_readed);
//////////////////////// Alpha expansion //////////////////////
        int num_labels = 3;
        int lambada=.45*255;
        Mat downSmp;
        resize(outImage, downSmp, Size(),(double)1/p_size, (double)1/p_size, INTER_NEAREST);
                // smoothness and data costs are set up one by one, individually
        enerfyMin = GridGraph_Individually(num_labels,downSmp,lambada);
        // resize as the size of the original image
        resize(enerfyMin, enerfyMin, image.size(), INTER_NEAREST);
        Mat enerfyMinSplit[3];
        split(enerfyMin, enerfyMinSplit);
        for (int i=0;i<3;i++)
                threshold(enerfyMinSplit[i],enerfyMinSplit[i],125,255,THRESH_BINARY);
        merge(enerfyMinSplit,3,enerfyMin);
        return enerfyMin;

}



Mat crtTestFet(Mat& image, char *model_readed)
{

        Mat locbp = makeLbpImg(image);
        // calling patch listing function---
        Rect imageprp(p_size,0,image.cols,image.rows);
        Mat listofpatch = patchpos(imageprp);

        Mat outImage(locbp.rows,locbp.cols, CV_8UC3, Scalar(0,0,0));
    /// Establish the number of bins
        int histSize = 256;
        float range[] = { 0, 256 } ; //the upper boundary is exclusive
        const float* histRange = { range };
        Mat hist;
        bool uniform = true; bool accumulate = false;
        Mat patch;
        int psiz = listofpatch.at<float>(1,2)-listofpatch.at<float>(0,2);
        //cout<<"====="<<psiz<<endl;
        CvSVM svm;
        svm.load(model_readed);
        for (int i=0;i<listofpatch.rows;i++)//
        {

                Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
                // Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
                //cout<<"====="<<ross<<", i= "<<i<<endl;
                patch= locbp(ross);
                // calculating histogram
                calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
                //hist = hist/(psiz*psiz);
                //hconcat(outFeture.feature,hist,outFeture.feature);
                float response = svm.predict(hist);
                if (response==0)
                        rectangle(outImage, ross, Scalar(255,0,0), -1, 8, 0 );
                if (response==1)
                        rectangle(outImage, ross, Scalar(0,255,0), -1, 8, 0 );
                if (response==2)
                        rectangle(outImage, ross, Scalar(0,0,255), -1, 8, 0 );

                //cout<<"predicted values = "<<response<<endl;
                //hconcat(outFeture.rectBox,ross,outFeture.rectBox);
        }

        return outImage;
}

```
