#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "lbp.hpp"
#include "histogram.hpp"
#include "GCoptimization.h"
#include "LinkedBlockList.h"
#include <errno.h>
#include<stdio.h>
#include<cstdlib>
#include<string.h>
#include<fstream>
#include<sstream>
#include<dirent.h>
#include "functionMain.h"
#include<dirent.h>
#include <string> 
#include <math.h>
#include <caffe/caffe.hpp>
//#include "generic.h"
#include <vl/kmeans.h>
#include <vl/lbp.h>
#include "rectprior.h"

using namespace cv;
using namespace std;
//-----------------------------------
using namespace caffe; 
using std::string;

//-----------------------------------------------------------------------------------
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
Mat Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);
Mat deepFet(output.size(),1,CV_32F,Scalar(0));
	for(int i=0;i<output.size();i++)
		deepFet.at<float>(i,0) = output[i];
	//std::cout<<"size = "<<output.size()<<std::endl;
 /* N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
  predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }
*/
  return deepFet;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->ForwardPrefilled();

  /* Copy the output layer to a std::vector */
  //Blob<float>* output_layer = net_->output_blobs()[0];
  const shared_ptr< Blob< float > > output_layer = net_->blob_by_name("fc7");
  //data, conv1, pool1, norm1, conv2, pool2, norm2, conv3, conv4, conv5, pool5, fc6, fc7, fc8,
  
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, CV_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, CV_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}










//------------------------------------------------------------------------------------------------
bool momentFet = false;



int countWhite(Mat src)
{
	int count_white=0;
	for( int y = 0; y < src.rows; y++ ) 
  		for( int x = 0; x < src.cols; x++ ) 
    			if ( src.at<uchar>(y,x) != 0 ) 
        			count_white++;
	return count_white;
}

float *findFmeasure(Mat outImg, Mat gtImg, int id)
{


		// split in to channels
		Mat outSplit[3],gtSplit[3]; // 0--> text region, 1--> graphics region, 2--> background

		split(outImg, outSplit);
		split(gtImg, gtSplit);
		// for checking purpose
//gtSplit[0] = Mat::zeros(outImg.rows,outImg.cols, CV_8UC1);
//outSplit[0] = Mat::zeros(outImg.rows,outImg.cols, CV_8UC1);
		// finding the common area
    		Mat andded[3];
		static float precision[3]={0.0,0.0,0.0},recall[3]={0.0,0.0,0.0},fscore[3]={0.0,0.0,0.0},delta=0.01;
		
		// to remove the blour in the output image
		for (int i=0;i<3;i++)
		{
			threshold(outSplit[i],outSplit[i],125,255,THRESH_BINARY);
			threshold(gtSplit[i],gtSplit[i],125,255,THRESH_BINARY);
			if (countWhite(gtSplit[i]) ==0)
			{
				precision[i] = -1;
				recall[i] = -1;
				fscore[i] = -1;		
			}
			else if (countWhite(gtSplit[i]) !=0 and countWhite(outSplit[i]) ==0)
			{
				precision[i] = 0.0;
				recall[i] = 0.0;
				fscore[i] = 0.0;
			}
			else
			{
				
    				bitwise_and(outSplit[i],gtSplit[i],andded[i]);
				precision[i] = (float)countWhite(andded[i])/(countWhite(outSplit[i])+delta);
				recall[i] = (float)countWhite(andded[i])/countWhite(gtSplit[i]);
				fscore[i] = (float)2*(precision[i]*recall[i])/(precision[i]+recall[i]+delta);
			}

		}
		//fscore[3] = recall[0];
	if(id==1)	
	return fscore;
	else if(id==2)	
	return precision;
	else if(id==3)	
	return recall;	
	else return 0;
}

void creatingVisul(char *orgImgF,char *maskF, char *resultF)
{
	Mat outImg, mask;
	vector<string> files = vector<string>();
	getdir(maskF,files);//outPut image mask folder
	for (unsigned int i = 0;i < files.size();i++) 
	{
		try 
		{
			string outImgPath =  orgImgF + string("/") + files[i] ;
			string gtImgPath =  maskF + string("/") + files[i] ;
			string outPath = resultF+ string("/") + files[i] +string(".png") ;
			
			cout<<"Now running "<<files[i]<<endl;
	    		outImg = imread(outImgPath, CV_LOAD_IMAGE_COLOR);	
			mask = imread(gtImgPath, CV_LOAD_IMAGE_COLOR);	
			Mat blendImg;
			addWeighted(outImg,0.5,mask,.5,0.0,blendImg);
			imwrite(outPath,blendImg);
		}
		catch( const std::exception& e ) {
    			 std::cout << e.what(); }
    	}		 
}


void CreateFineTuneGt(char *orgImgF,char *maskF, char *resultF)
{
	Mat outImg, mask;
	vector<string> files = vector<string>();
	getdir(maskF,files);//outPut image mask folder
	for (unsigned int i = 0;i < files.size();i++) 
	{
		try 
		{
			string outImgPath =  orgImgF + string("/") + files[i] ;
			string gtImgPath =  maskF + string("/") + files[i] ;
			string outPath = resultF+ string("/") + files[i] ;
			
			cout<<"Now running "<<files[i]<<endl;
	    		outImg = imread(outImgPath, 0);
	    			
			mask = imread(gtImgPath, CV_LOAD_IMAGE_COLOR);	
			threshold(outImg,outImg, 0, 255, CV_THRESH_OTSU);

			Mat gtsplit[3],merged;
			split(mask, gtsplit);
			for (int j=0;j<3;j++)
				threshold(gtsplit[j],gtsplit[j],125,255,THRESH_BINARY);
				
			bitwise_not(outImg,outImg);
			Mat graNot, textNot;			
			bitwise_not(gtsplit[1],graNot);
			bitwise_and(gtsplit[0],outImg,gtsplit[0] );
			bitwise_not(gtsplit[0],textNot);
			bitwise_and(graNot,textNot,gtsplit[2] );
	
			merge(gtsplit,3,merged);
			//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
			//imshow( "Display window", merged ); 
			//waitKey(0);	

			imwrite(outPath,merged);
		}
		catch( const std::exception& e ) {
    			 std::cout << e.what(); }
    	}		 
}

void evaluateOutput(char *outFolder,char *gtFolder, char *resultFileName)
{
Mat outImg, gtImg;
// reading all files in the folder
		vector<string> files = vector<string>();
		getdir(outFolder,files);//outPut image folder
		
	  	ofstream resultOut;

		resultOut.open (resultFileName,ios::app);
		int count0=0,count1=0,count2=0;
		float val0=0,val1=0,val2=0; 
		for (unsigned int i = 0;i < files.size();i++) 
		{
		try {
			string outImgPath =  outFolder + string("/") + files[i] ;
			string gtImgPath =  gtFolder + string("/") + files[i] ;
			cout<<"Now running "<<files[i]<<endl;


	    		outImg = imread(outImgPath, CV_LOAD_IMAGE_COLOR);	
			gtImg = imread(gtImgPath, CV_LOAD_IMAGE_COLOR);

			float *fmesurre;
			fmesurre=  findFmeasure (outImg,gtImg,1);
			for (int j=0; j<3;j++)
				cout<<"kk "<<*(fmesurre+j)<<endl;
			resultOut<<files[i]<<"\t"<<*(fmesurre+0)<<"\t"<<*(fmesurre+1)<<"\t"<<*(fmesurre+2)<<"\n";
			// to get the average value///
			if (*(fmesurre+0)>=0)
			{
				count0++;
				val0=val0+*(fmesurre+0);	
			}
			if (*(fmesurre+1)>=0)
			{
				count1++;
				val1=val1+*(fmesurre+1);	
			}
			if (*(fmesurre+2)>=0)
			{
				count2++;
				val2=val2+*(fmesurre+2);	
			}
		}
		catch( const std::exception& e ) {
    			 std::cout << e.what(); }
	}
	float avgfm[3]={0,0,0};
	avgfm[0] = (float)val0/(count0+.0001);
	avgfm[1] = (float)val1/(count1+.0001);
	avgfm[2] = (float)val2/(count2+.0001);
	resultOut<<"Average "<<"\t"<<avgfm[0]<<"\t"<<avgfm[1]<<"\t"<<avgfm[2]<<"\n";
	resultOut.close();

}

//------------------------------------

void newEvaluateOutput(char *outFolder,char *gtFolder, char *resultFileName) // adding recall at the end
{
Mat outImg, gtImg;
// reading all files in the folder
		vector<string> files = vector<string>();
		getdir(outFolder,files);//outPut image folder
		
	  	ofstream resultOut;

		resultOut.open (resultFileName,ios::app);
		int count0=0,count1=0,count2=0,count3=0;
		float val0=0, val1=0, val2=0, val10=0, val11=0, val12=0, val20=0, val21=0, val22=0; 
		for (unsigned int i = 0;i < files.size();i++) 
		{
		try {
			string outImgPath =  outFolder + string("/") + files[i] ;
			string gtImgPath =  gtFolder + string("/") + files[i] ;
			cout<<"Now running "<<files[i]<<endl;


	    		outImg = imread(outImgPath, CV_LOAD_IMAGE_COLOR);	
			gtImg = imread(gtImgPath, CV_LOAD_IMAGE_COLOR);

			float *fmesurre, *pricitions, *recalls;
			fmesurre= findFmeasure(outImg,gtImg,1);// 1- F->measure, 2-> pricition, 3->Recall
			pricitions = findFmeasure(outImg,gtImg,2);
			recalls = findFmeasure(outImg,gtImg,3);
			//for (int j=0; j<4;j++)
				//cout<<"kk "<<*(fmesurre+j)<<endl;
			//resultOut<<files[i]<<"\t"<<*(fmesurre+0)<<"\t"<<*(fmesurre+1)<<"\t"<<*(fmesurre+2)<<"\t"<<*(fmesurre+3)<<"\n";
			// to get the average value///
			if (*(fmesurre+0)>=0){
			count0++;
			val0=val0+*(fmesurre+0);
			val10=val10+*(pricitions+0);
			val20=val20+*(recalls+0);
			}
			if (*(fmesurre+1)>=0){
			count1++;
			val1=val1+*(fmesurre+1);
			val11=val11+*(pricitions+1);
			val21=val21+*(recalls+1);
			}
			if (*(fmesurre+2)>=0){
			count2++;
			val2=val2+*(fmesurre+2);
			val12=val12+*(pricitions+2);
			val22=val22+*(recalls+2);
			}
			
		}
		catch( const std::exception& e ) {
    			 std::cout << e.what(); }
	}
	float avgfm[3]={0,0,0};
	avgfm[0] = (float)val0/(count0+.0001);
	avgfm[1] = (float)val1/(count1+.0001);
	avgfm[2] = (float)val2/(count2+.0001);
	resultOut<<" -----------"<<"\t"<<" Text "<<"\t"<<"\t"<<" Graphics "<<"\t"<<" Background "<<"\t|"<<"\n";
	resultOut<<" ----------------------------------------------------------------\n";
	resultOut<<"f-measure "<<"\t"<<avgfm[0]<<"\t"<<avgfm[1]<<"\t"<<avgfm[2]<<"\t|"<<"\n";
	float avgfm1[3]={0,0,0};
	avgfm1[0] = (float)val10/(count0+.0001);
	avgfm1[1] = (float)val11/(count1+.0001);
	avgfm1[2] = (float)val12/(count2+.0001);
	resultOut<<"Precition "<<"\t"<<avgfm1[0]<<"\t"<<avgfm1[1]<<"\t"<<avgfm1[2]<<"\t|"<<"\n";
	float avgfm2[3]={0,0,0};
	avgfm2[0] = (float)val20/(count0+.0001);
	avgfm2[1] = (float)val21/(count1+.0001);
	avgfm2[2] = (float)val22/(count2+.0001);
	resultOut<<"Recall "<<"\t"<<"\t"<<avgfm2[0]<<"\t"<<avgfm2[1]<<"\t"<<avgfm2[2]<<"\t|"<<"\n";		
	resultOut.close();

}

//============================================

void textDet(Mat image)
{
	Mat binImg, gray;
	cout<<"image type "<<image.type()<<endl;
	if (image.type()==16 or image.type()==24)
	{
	cvtColor(image, gray, CV_RGB2GRAY);
	cout<<"Color image"<<endl;

	
	//threshold(gray,binImg, 0, 255, CV_THRESH_OTSU);
	//bitwise_not ( binImg, binImg );
	
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", binImg ); 
	//waitKey(0);
	//int info = textdet(binImg, 200);
	//if (info)
	//cout<<"tetect graphics"<<endl;
	//else
	//{
	//cout<<"tetect pure txt"<<endl;
	//ofstream resultOut;
	//resultOut.open ("pureTxt.txt",ios::app);
	//resultOut<<"pure txt "<<"\n";
	//resultOut.close();	
	//}
	//===========================================
  Mat src, dst;

  /// Load image
  src =image.clone();

  /// Separate the image in 3 places ( B, G and R )
  vector<Mat> bgr_planes;
  split( src, bgr_planes );

  /// Establish the number of bins
  int histSize = 256;

  /// Set the ranges ( for B,G,R) )
  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat b_hist, g_hist, r_hist;

  /// Compute the histograms:
  calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

	//cout<<"b_hist.size() = "<<sum(b_hist).val[0]<<endl;
	Mat sub_mat_1 = b_hist(cv::Range(254, 256),   cv::Range::all());
	Mat sub_mat_11 = b_hist(cv::Range(10, 220),   cv::Range::all());
	//cout<<sub_mat_1<<endl;
	//cout<<"sub_mat_1 sum = "<<sum(sub_mat_1).val[0]<<endl;
	
	//cout<<"g_hist.size() = "<<sum(g_hist).val[0]<<endl;
	Mat sub_mat_2 = g_hist(cv::Range(254, 256),   cv::Range::all());
	Mat sub_mat_22 = g_hist(cv::Range(10, 220),   cv::Range::all());
	//cout<<"sub_mat_2 sum = "<<sum(sub_mat_1).val[0]<<endl;	
	
	//cout<<"r_hist.size() = "<<sum(r_hist).val[0]<<endl;
	Mat sub_mat_3 = r_hist(cv::Range(254, 256),   cv::Range::all());
	Mat sub_mat_33 = r_hist(cv::Range(10, 220),   cv::Range::all());
	//cout<<"sub_mat_3 sum = "<<sum(sub_mat_3).val[0]<<endl;
	
	int dif = abs(sum(sub_mat_1).val[0]-sum(sub_mat_2).val[0])+abs(sum(sub_mat_1).val[0]-sum(sub_mat_3).val[0])+abs(sum(sub_mat_3).val[0]-sum(sub_mat_2).val[0]);
	int summn = sum(sub_mat_11).val[0]+sum(sub_mat_22).val[0]+sum(sub_mat_33).val[0];
	//cout<<"abs = "<<dif<<endl;
	if (dif>=11000 and dif>=10 and summn<=100)
	cout<<"Detect graphics"<<endl;
	else
	{
	cout<<"Detect pure txt"<<endl;
	ofstream resultOut;
	resultOut.open ("pureTxt.txt",ios::app);
	resultOut<<"pure txt "<<dif<<"\n";
	resultOut.close();	
	}
	}
   /* */
	//cout<<"info = "<<info<<endl;
}

//int p_size = 50;
// FUNCTION FROM ALPHA EXPANSION

//struct ForDataFn{
//	int numLab;
//	int *data;
//};


int smoothFn(int p1, int p2, int l1, int l2)
{
	if ( (l1-l2)*(l1-l2) <= 4 ) return((l1-l2)*(l1-l2));
	else return(4);
}

int dataFn(int p, int l, void *data)
{
	ForDataFn *myData = (ForDataFn *) data;
	int numLab = myData->numLab;

	return( myData->data[p*numLab+l] );
}



////////////////////////////////////////////////////////////////////////////////
// smoothness and data costs are set up one by one, individually
// grid neighborhood structure is assumed
//
Mat GridGraph_Individually(int num_labels,Mat img,int lambda)
{

	int height=img.rows;//HEIGHT
	int width=img.cols;//width
	int num_pixels=height*width;

	int *result = new int[num_pixels];   // stores result of optimization
	int rw;
	int col;
	Mat  opimage =img.clone();
//image is transformed int 1 drow in row major order

	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// first set up data costs individually


		for ( int i = 0; i < num_pixels; i++ )
		{
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;

			}	
			else
			{
			rw=(i+1)/width;
			col=((i+1)%width)-1;
			}

			int blue=img.at<cv::Vec3b>(rw,col)[0];
			int green=img.at<cv::Vec3b>(rw,col)[1];
			int red=img.at<cv::Vec3b>(rw,col)[2];



			for (int l = 0; l < num_labels; l++ )
			{
				if(l==0)
					 gc->setDataCost(i,l,(255-blue)/*+red+green*/);
			 	if(l==1)
			 		gc->setDataCost(i,l,(255-green)/*+red+blue*/);
		 		if(l==2)
		 			gc->setDataCost(i,l,(255-red)/*+blue+green*/);

			}
		}

		// next set up smoothness costs individually
		for ( int l1 = 0; l1 < num_labels; l1++ )
			for (int l2 = 0; l2 < num_labels; l2++ )
			{

				if(l1==l2)
				//int cost = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;
				gc->setSmoothCost(l1,l2,0);

				else

				gc->setSmoothCost(l1,l2,lambda);


			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());


		

		for ( int  i = 0; i < num_pixels; i++ )
		{
			result[i] = gc->whatLabel(i);
			if((i+1)%width==0 )
			{
				rw=((i+1)/width)-1;
				col=width-1;
			}
			else
			{
				rw=(i+1)/width;
				col=((i+1)%width)-1;
			}
			if(result[i]==0) //sky
			{
		//cout<<"label 0 \n";
				opimage.at<cv::Vec3b>(rw,col)[0]=255;//blue
				opimage.at<cv::Vec3b>(rw,col)[1]=0;
				opimage.at<cv::Vec3b>(rw,col)[2]=0;
			}
			if(result[i]==1) // grass
			{
			opimage.at<cv::Vec3b>(rw,col)[0]=0;
			opimage.at<cv::Vec3b>(rw,col)[1]=255;
			opimage.at<cv::Vec3b>(rw,col)[2]=0;
			//cout<<"label 1 \n";
			}
			if(result[i]==2) //third object
			{
				opimage.at<cv::Vec3b>(rw,col)[0]=0;
				opimage.at<cv::Vec3b>(rw,col)[1]=0;
				opimage.at<cv::Vec3b>(rw,col)[2]=255;//red
			}
		}





		//imwrite( "outputimage.png", opimage );


		delete gc;
	}
	catch (GCException e)
	{
		e.Report();
	}
	delete [] result;
	return opimage;
}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
void GridGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);
		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
//
void GridGraph_DfnSfn(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);

		// set up the needed data to pass to function for the data costs
		ForDataFn toFn;
		toFn.data = data;
		toFn.numLab = num_labels;

		gc->setDataCost(&dataFn,&toFn);

		// smoothness comes from function pointer
		gc->setSmoothCost(&smoothFn);

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
// Uses spatially varying smoothness terms. That is
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1
void GridGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;

	// next set up spatially varying arrays V and H

	int *V = new int[num_pixels];
	int *H = new int[num_pixels];


	for ( int i = 0; i < num_pixels; i++ ){
		H[i] = i+(i+1)%3;
		V[i] = i*(i+width)%7;
	}


	try{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(width,height,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCostVH(smooth,V,H);
		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}

////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually"
//
void GeneralGraph_DArraySArray(int width,int height,int num_pixels,int num_labels)
{

	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ )
				gc->setNeighbors(x+y*width,x-1+y*width);

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ )
				gc->setNeighbors(x+y*width,x+(y-1)*width);

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;

}
////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood is set up "manually". Uses spatially varying terms. Namely
// V(p1,p2,l1,l2) = w_{p1,p2}*[min((l1-l2)*(l1-l2),4)], with
// w_{p1,p2} = p1+p2 if |p1-p2| == 1 and w_{p1,p2} = p1*p2 if |p1-p2| is not 1

void GeneralGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 5 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ ){
				int p1 = x-1+y*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1+p2);
			}

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ ){
				int p1 = x+(y-1)*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1*p2);
			}

		//printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		//printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ )
			result[i] = gc->whatLabel(i);

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}

////////////////////////////////////////////





Mat patchpos(Rect imageprp)
{
// this function will give the starting xy position of all aptch from the given image
	//int p_size = imageprp.x;
	int nobk=0;
	nobk = floor( ((double) imageprp.height/p_size))*floor( ((double) imageprp.width/p_size)) +floor( ((double) imageprp.height/p_size))+floor( ((double) imageprp.width/p_size))+1;
	
	// cout<<" currect no p ="<<floor( ((double) imageprp.height/p_size))*floor( ((double) imageprp.width/p_size))<<endl;

	// cout<<" total p= "<<nobk<<endl;
	// cout<<"height= "<<imageprp.height<<endl;
	// cout<<"width= "<<imageprp.width<<endl;
	Mat codi_list(nobk,3, CV_32F,Scalar(0));
	int cnt=0;
	for (int i =0;i<imageprp.height-1-p_size;i=i+p_size)//Row
	{
		for (int j =0;j<imageprp.width-1-p_size;j=j+p_size)//Col
		{
			codi_list.at<float>(cnt,0)=cnt;
			codi_list.at<float>(cnt,1)=i;
			codi_list.at<float>(cnt,2)=j;
			cnt=cnt+1;
		}
	}
	//cout<<"1st "<<cnt<<endl;
	for (int i =0;i<imageprp.height-1-p_size;i=i+p_size)//Row
	{

		int j=imageprp.width-2-p_size; // -2 because the lbp function gives out image 2 row/col less than the original
		codi_list.at<float>(cnt,0)=cnt;
		codi_list.at<float>(cnt,1)=i;
		codi_list.at<float>(cnt,2)=j;
		cnt=cnt+1;
	}	
	//cout<<"2nd "<<cnt<<endl;
	for (int j =0;j<imageprp.width-1-p_size;j=j+p_size)//Row
	{

		int i=imageprp.height-2-p_size;
		codi_list.at<float>(cnt,0)=cnt;
		codi_list.at<float>(cnt,1)=i;
		codi_list.at<float>(cnt,2)=j;
		cnt=cnt+1;

	}

	//cout<<"3rd "<<cnt<<endl;
	codi_list.at<float>(cnt,0)=cnt;
	codi_list.at<float>(cnt,1)=imageprp.height-p_size-2;
	codi_list.at<float>(cnt,2)=imageprp.width-p_size-2;
	//cout<<"4th "<<cnt<<endl;
	return codi_list;

/*	usage of this function listofpatch = patchpos(imageprp);

Ploting all patches boundary on image.
	for (int i=0;i<listofpatch.rows;i++)
	{
	int psiz = listofpatch.at<float>(1,0)-listofpatch.at<float>(0,0);
	Rect roi(listofpatch.at<float>(i,0),listofpatch.at<float>(i,1),psiz,psiz);
	rectangle(image, roi, Scalar(255*rand(),255*rand(),255*rand()), 1, 8, 0 );	
	}
*/
}

//============***************** ctreating training features*****************////////
ThreeMatstr crtTrainFet(Mat& listofpatch,Mat& locbp,Mat& gt_img)
{

	ThreeMatstr outFeture;
	Mat initial(256,1,CV_32F,Scalar(0));
	//cout<<"text size = "<<initial.size()<<endl;
	outFeture.background = initial.clone();
	outFeture.figure	= initial.clone();
	outFeture.text		= initial.clone();
    /// Establish the number of bins
	int histSize = 256;
	float range[] = { 0, 256 } ; //the upper boundary is exclusive
	const float* histRange = { range };
	Mat hist;
	bool uniform = true; bool accumulate = false;
	Mat fetr_lst, main_lst;
	Mat patch,label,channel[3],label_list(listofpatch.rows,2, CV_32F,Scalar(0)), lbl(1,1, CV_32F,Scalar(0));
	int psiz = listofpatch.at<float>(1,2)-listofpatch.at<float>(0,2);
	//cout<<"====="<<psiz<<endl;

	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
		patch= locbp(ross);
//	--------creating labels for each patch-----------------------
		label = gt_img(ross);
		split(label, channel);
		// calculating histogram
		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		//calculating label
		//hist = hist/(psiz*psiz);
		//cout<<"hist = "<<hist<<endl;
		double sum_b = cv::sum( channel[0] )[0];// text region 			=>label 0
		double sum_g = cv::sum( channel[1] )[0];// figure region		=>label 1
		double sum_r = cv::sum( channel[2] )[0];// background region	=>label 2
		//Assigning to curresponding lablel
		if (sum_b>=sum_g and sum_b>=sum_r)
		{
			hconcat(outFeture.text,hist,outFeture.text);
			// cout<<"text size = "<<outFeture.text.size()<<endl;
		}
		if (sum_g>=sum_b and sum_g>=sum_r)

		{
			hconcat(outFeture.figure,hist,outFeture.figure);
			// cout<<"figure size = "<<outFeture.figure.size()<<endl;
		}
		if (sum_r>=sum_b and sum_r>=sum_g)
			
		{
			hconcat(outFeture.background,hist,outFeture.background);
			// cout<<"background size = "<<outFeture.background.size()<<endl;
		}
		
		
		//hconcat(lbl, hist.t(), fetr_lst);

		//fetr_lst = hist.clone();
		//if (i==0)
		//{
		//	outFeture.background = hist.clone();
		//	continue;
		//}


	}
	
	return outFeture;
}
Mat makeLbpImg(Mat& image)
{
// ------------------Taking lbp from the image--------------------------------------
    // initial values
	int radius = 1;
	int neighbors = 8;
	Mat dst,locbp; 
    // matrices used


    // just to switch between possible lbp operators
/*  vector<string> lbp_names;
    lbp_names.push_back("Extended LBP"); // 0
    lbp_names.push_back("Fixed Sampling LBP"); // 1
    lbp_names.push_back("Variance-based LBP"); // 2
    int lbp_operator=0;*/
// pre-processing part
    cvtColor(image, dst, CV_BGR2GRAY);
	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    // -----------------------different lbp functions-----------------------------------
    //lbp::ELBP(dst, locbp, radius, neighbors); // use the extended operator
	lbp::OLBP(dst, locbp); // use the original operator
    	//lbp::VARLBP(dst, locbp, radius, neighbors);
    // a simple min-max norm will do the job...
	normalize(locbp, locbp, 0, 255, NORM_MINMAX, CV_8UC1);
//-----------------------geting patch list ------------------------------------
	return locbp;

}
//@@@@@@@@@@@@@@@@@@@@@ read all file from the folder@@@@@@@@@@@@@@@@@@@@@@


int getdir (string dir, vector<string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if((dp  = opendir(dir.c_str())) == NULL) 
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) 
	{
		if( strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0 )
		{
    		//cout<<dirp->d_name<<"hohohhhh"<<endl;
			files.push_back(string(dirp->d_name));	
		}

	}
	closedir(dp);
	return 0;
}
//!!!!!!!!!!!!!!!!!!!!Cleaning feature vector!!!!!!!!!!!++++++++++++@@@@@@@@@@@@@
Mat RmZeroCols(Mat input)
{
	Mat outMat,temp;
	outMat = input.col(10);
	//temp = train_feture.text.col(10);
	//cout<<"dimentios= "<<input.cols<<endl;
	for (int i = 0; i < input.cols; ++i)
	{
		temp = input.col(i);
	// cout<<"dimentios= "<<train_feture.text.col(1)<<endl;
		//cout<<"dimentios= "<<sum(temp)[0]<<endl;
		if(sum(temp)[0]!=0)
		{
			hconcat(outMat,temp,outMat);
		}
		//else
			//cout<<"Skipped ="<<i<<endl;
		//cout<<"dimentios= "<<cleaned.size()<<""<<i<<endl;
	}
	return outMat;

}

ThreeMatstr cleanFet(ThreeMatstr train_feture)
{
	ThreeMatstr clean_feture;
	clean_feture.text = RmZeroCols(train_feture.text);
	clean_feture.figure = RmZeroCols(train_feture.figure);
	clean_feture.background = RmZeroCols(train_feture.background);
	// make equal dimension
		// cout<<"dimentios of clean_feture.text= "<<clean_feture.text.cols<<endl;
		// cout<<"dimentios of clean_feture.figure= "<<clean_feture.figure.cols<<endl;
		// cout<<"dimentios of clean_feture.background= "<<clean_feture.background.cols<<endl;
		// int mmin = std::min(clean_feture.text.cols, clean_feture.figure.cols);
		// mmin = std::min(clean_feture.background.cols, mmin);
		// cout<<"mim = "<<mmin<<endl;
	//cout<<"dimentios last= "<<clean_feture.text.size()<<endl;
	return clean_feture;

}
/* Function maximum definition */
/* x, y and z are parameters */
int maximum(int x, int y, int z) {
	int max = x; /* assume x is the largest */

	if (y > max) { /* if y is larger than max, assign y to max */
		max = y;
	} /* end if */

	if (z > max) { /* if z is larger than max, assign z to max */
	max = z;
	} /* end if */

	return max; /* max is the largest value */
} /* end function maximum */
//############### SVM training function ###################
void TrainTheModel(string org_folder,string gt_folder, char *model_name)
{
		ThreeMatstr finalFet;
//		Mat initial(256,1,CV_32F,Scalar(0));
//	//cout<<"text size = "<<initial.size()<<endl;
//		finalFet.background = initial.clone();
//		finalFet.figure		= initial.clone();
//		finalFet.text		= initial.clone();
		vector<ThreeMatstr>  feturelist;
		vector<string> files = vector<string>();
		getdir(org_folder,files); 
//	final feture list
		Mat final_lst;
		//cout<<"testt= "<<files.size()<<endl;
                ::google::InitGoogleLogging("./docSeg");
                string model_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/deploy.prototxt";
                string trained_file = "/users/jobinkv/installs/caffe_cpp/googleNet/bvlc_reference_caffenet.caffemodel";
                string mean_file    = "/users/jobinkv/installs/caffe_cpp/googleNet/imagenet_mean.binaryproto";
                string label_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/synset_words.txt";
                Classifier deepFt(model_file, trained_file, mean_file, label_file);

		for (unsigned int i = 0;i < files.size();i++) 
		{
			 if (i>=3)// looop for sample run
			 	break;
			string gt_path =  gt_folder + string("/") + files[i] ;
			string org_path = org_folder + string("/") + files[i] ;

			Mat image, gt_img;
	    		image = imread(org_path, CV_LOAD_IMAGE_COLOR);   // Reading original image
			gt_img = imread(gt_path, CV_LOAD_IMAGE_COLOR);   // Reading gt image
		cout<<"Entered!!!!!!"<<endl;
			//--------------------------------------------------------------------
			resize(gt_img, gt_img, image.size(), INTER_NEAREST);
			Mat enerfyMinSplit[3];
			split(gt_img, enerfyMinSplit);
			for (int j=0;j<3;j++)
				threshold(enerfyMinSplit[j],enerfyMinSplit[j],125,255,THRESH_BINARY);
			merge(enerfyMinSplit,3,gt_img);
		//	imwrite(gt_path,gt_img);
		//	continue;
			
			//--------------------------------------------------------------------
			if( gt_img.cols != image.cols or gt_img.rows != image.rows )
			{
				cout <<"ERROr : Image dimentios of the given images "<<files[i]<<" are not matching" << endl;
				continue;
			}
			cout<<"Now running "<<files[i]<<endl;
		//	Rect imageprp(p_size,0,image.cols,image.rows); 
		//	Mat locbp = makeLbpImg(image);
		//	Mat listofpatch = patchpos(imageprp);
		//	ThreeMatstr train_feture = crtTrainFet(listofpatch,locbp,gt_img);
			ThreeMatstr train_feture = crtTrainFetGabur(image,gt_img, deepFt);
			feturelist.push_back(train_feture);
			cout<<"feature size  = "<<train_feture.text.size()<<endl;
		// concatinating the out put features
			//hconcat(finalFet.text,train_feture.text,finalFet.text);
			//hconcat(finalFet.figure,train_feture.figure,finalFet.figure);
			//hconcat(finalFet.background,train_feture.background,finalFet.background);
		}
		// ThreeMatstr clean_feture = cleanFet(finalFet);
     
               /*  //ThreeMatstr clean_feture = list2mat1(feturelist);

                Mat textFeture1 = L2Normalization(clean_feture.text.t());
                Mat figue1 =L2Normalization(clean_feture.figure.t());
                Mat backGnd1 = L2Normalization(clean_feture.background.t());
		cout<<textFeture1.size()<<endl;	
		cout<<figue1.size()<<endl;	
		cout<<backGnd1.size()<<endl;	




*/

//--------------------------------------------------------
           ThreeMatstr clean_fetures = list2mat1(feturelist);

                Mat textFeture = L2Normalization(clean_fetures.text);
                Mat figue =L2Normalization(clean_fetures.figure);
                Mat backGnd = L2Normalization(clean_fetures.background);
		cout<<"text size = "<<textFeture.size()<<endl;
		cout<<"figue size = "<<figue.size()<<endl;
		cout<<"backGnd size = "<<backGnd.size()<<endl;
		//ThreeMatstr clean_feture = finalFet;
		int maxx = maximum(textFeture.cols, figue.cols, backGnd.cols);
		cout<<"maximum val= "<<maxx<<endl;
		cout<<"text size= "<<textFeture.cols<<endl;
		cout<<"figure size= "<<figue.cols<<endl;
		cout<<"background size= "<<backGnd.cols<<endl;
		Mat trainData;
		hconcat(textFeture, figue,trainData);
		hconcat(trainData,backGnd,trainData);
	// making of labels
		Mat labels;
		Mat lab_text(textFeture.cols,1,CV_32F,Scalar(0));
		Mat lab_figure(figue.cols,1,CV_32F,Scalar(1));
		Mat lab_background(backGnd.cols,1,CV_32F,Scalar(2));
		vconcat(lab_text,lab_figure,labels);
		vconcat(labels,lab_background,labels);
		trainData = trainData.t();
		cout<<"trainData.size() = "<<trainData.size()<<endl;
		cout<<"labels.size() = "<<labels.size()<<endl;
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
		param.C = 1; // for CV_SVM_C_SVC , CV_SVM_EPS_SVR  and  CV_SVM_NU_SVR
		param.nu = 0.0; // for  CV_SVM_NU_SVC, CV_SVM_ONE_CLASS , and  CV_SVM_NU_SVR
		param.p = 0.0; // for CV_SVM_EPS_SVR
		param.class_weights = &weights;//[(.6, 0.3,0.1);//NULL;//for CV_SVM_C_SVC
		param.term_crit.type = CV_TERMCRIT_ITER;	 //| CV_TERMCRIT_EPS;
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
		//svm.train(trainData, labels, Mat(), Mat(),param);
		svm.save(model_name);

		cout << "Finished training process" << endl;
}

//--------SVM testing patch creator---------------------//
Mat crtTestFet(Mat& image, char *model_readed,Classifier deep)
{
/*
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
*/	//cout<<"====="<<psiz<<endl;
	// new line added
	Mat gray;
        cvtColor(image,gray,CV_RGB2GRAY);
	fetExtrct meth1;
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
        Mat listofpatch = meth1.listOfpatch();

        Mat outImage(image.rows,image.cols, CV_8UC3, Scalar(0,0,0));
        Mat hist;
/*// deep classifier
        ::google::InitGoogleLogging("./docSeg");
        string model_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/deploy.prototxt";
        string trained_file = "/users/jobinkv/installs/caffe_cpp/googleNet/bvlc_reference_caffenet.caffemodel";
        string mean_file    = "/users/jobinkv/installs/caffe_cpp/googleNet/imagenet_mean.binaryproto";
        string label_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/synset_words.txt";
      	Classifier deepFt(model_file, trained_file, mean_file, label_file);
*/
		
//==========================
	CvSVM svm;
	svm.load(model_readed);
	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),p_size,p_size);
		/*// Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
		patch= locbp(ross);
		// calculating histogram
		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		//hist = hist/(psiz*psiz);
		*/ //hconcat(outFeture.feature,hist,outFeture.feature);
// new line added
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
//-----------------
		float response = svm.predict(L2Normalization(hist));
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

///////////////=======+++++++++++++ MAIN PROGRAM +++++++++++=============//////////////
Mat docLayotSeg(Mat image, char *model_readed,Classifier deep)
{

	Mat enerfyMin;
	Mat outImage = crtTestFet(image, model_readed, deep);
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
Mat rectPrior(Mat layout)
{

	// asuming reW=700, reH=500;
	int reW=2646-229, reH=1354-356;
	int imgHight=layout.rows,imgWidth = layout.cols;
	// making integral images
	Mat laySplit[3];
	split(layout, laySplit);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", laySplit[0] ); 
	//waitKey(0); 
	imwrite("testbg.png",laySplit[0]);
	// 
	Mat isumText, isumGraphics;
	for (int i=0;i<3;i++)
		threshold(laySplit[i],laySplit[i],125,1,THRESH_BINARY);	
	integral(laySplit[0],isumText,CV_64F);
	cout<<"input size = "<<laySplit[0].size()<<endl;
	cout<<"output size = "<<isumText.size()<<endl;
	
	integral(laySplit[1],isumGraphics,CV_64F);
	// making pixel value one

	
	Rect crntBk;
	crntBk.width =2646-229;crntBk.height=1354-356;
	

	 
	
	// for checking 
	//int i=229+100; int j=356+200;
	//int i=250,j=2279;
	reW=2373, reH=1000;
	//reW=9, reH=9;
	// analysis matrix
	Mat anals((imgWidth-reW),(imgHight-reH), CV_64F, Scalar(0));
	//
	for (int i=0;i<(imgWidth-reW);i++)//x
		for (int j=0;j<(imgHight-reH);j++)//y
		{
			double regionSum;
		      	double tl= isumText.at<double>(j,i);
		      	double tr= isumText.at<double>(j,(i+reW+1));
		      	double bl= isumText.at<double>((j+reH+1),i);
		      	double br= isumText.at<double>((j+reH+1),(i+reW+1));
		      	regionSum = br-bl-tr+tl;
		      	anals.at<double>(i,j)=regionSum;
			//cout<<"rect is "<<regionSum<<", i="<<i<<endl;
			//Mat temp = layout(crntBk).clone();
		}
	// maximum value
	Point min_loc, max_loc;
	double min, max;
	minMaxLoc(anals, &min, &max, &min_loc, &max_loc);
	cout<<"max_loc= "<<max_loc<<endl;
	Rect txtBk;
	txtBk.x=max_loc.y;
	txtBk.y=max_loc.x;
	txtBk.width = reW;
	txtBk.height = reH;
	// checking of integral image
	//double testOut = isumText.at<double>(i,j);
	//cout<<"tl haha="<<isumText.at<double>(j, i) <<endl;//(y,x)
	//VlKMeans * kmeans = vl_kmeans_new (VL_TYPE_FLOAT,VlDistanceL2) ;
	rectangle(layout,txtBk, Scalar(255,255,255),8);
		
	return layout;
}

Mat rectPrior3(Mat layout)
{
// without assuming width and height
	int imgHight=layout.rows,imgWidth = layout.cols;
// splitting the out put
	Mat laySplit[3];
	split(layout, laySplit);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", layout ); 
	//waitKey(0); 	
//making 1 and zero
	for (int i=0;i<3;i++)
		threshold(laySplit[i],laySplit[i],125,1,THRESH_BINARY);	
	Mat plotImg(imgHight, imgWidth, CV_8UC3,Scalar(0,0,255)); // for ploting purpose
	int cnt=0,delta=5;float costParam = .5;
// making of integral image for fast computation
	Mat isumText, isumGraphics;
	integral(laySplit[0],isumText,CV_64F);
	//integral(laySplit[1],isumGraphics,CV_64F);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", laySplit[0] ); 
	//waitKey(0); 	
//---------------------------------------------------------------------------------------------------------------------------------	
	bool iter=true;
	// for testing purpose
	//int j=400,i=350;
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if(laySplit[0].at<uchar>(j,i)==1)
			{
				Rect tempRect;
				tempRect.x=i;tempRect.y=j;tempRect.width=50;tempRect.height=50;
				// initial density check
				double initialC=0;
				double tl= isumText.at<double>(tempRect.y,tempRect.x);
				double tr= isumText.at<double>(tempRect.y,tempRect.x+tempRect.width);
				double bl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				initialC = br-bl-tr+tl;
				if (initialC<tempRect.width*tempRect.height)
					continue;// skip if the starting pixel is not valid
				br=0;bl=0;tr=0;tl=0;
				while(iter)
				{
					// finding the width cost
					double widthCost=0;
				      	double tl= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	double tr= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width+delta));
				      	double bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width+delta));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		widthCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	//finding height cost
				      	double heightCost=0;
				      	tl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	bl= isumText.at<double>((tempRect.y+tempRect.height+delta),tempRect.x);
				      	br= isumText.at<double>((tempRect.y+tempRect.height+delta),(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		heightCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding icost
				      	double icost=0;
				      	if (tempRect.x>delta){
				      	tl= isumText.at<double>(tempRect.y,(tempRect.x-delta));
				      	tr= isumText.at<double>(tempRect.y,tempRect.x);
				      	bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x-delta));
				      	br= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      	    	icost = br-bl-tr+tl;
				      	}else icost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding jcost
				      	double jcost=0;
				      	if (tempRect.y>delta){
				      	tl= isumText.at<double>((tempRect.y-delta),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y-delta),tempRect.x+tempRect.width);
				      	bl= isumText.at<double>(tempRect.y,tempRect.x);
				      	br= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)				      						      	jcost = br-bl-tr+tl;
				      	}else jcost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	if (jcost>=costParam*tempRect.width*delta)
				      		{
				      		tempRect.y=tempRect.y-delta;
				      		tempRect.height=tempRect.height+delta;
				      		}
				      	if (icost>=costParam*tempRect.height*delta)
				      		{
				      		tempRect.x=tempRect.x-delta;
				      		tempRect.width=tempRect.width+delta;
				      		}
				      	if (widthCost>=costParam*tempRect.height*delta)
				      		tempRect.width=tempRect.width+delta;
				      	if (heightCost>=costParam*tempRect.width*delta)
				      		tempRect.height=tempRect.height+delta;
				      	// condition to exit the loop
				      	if(icost<costParam*tempRect.height*delta and widthCost<costParam*tempRect.height*delta and heightCost<costParam*tempRect.width*delta and jcost<costParam*tempRect.width*delta)
				      		iter=false;
			      	}

			      	//cout<<"the rect width is = "<<tempRect.width<<endl;
			      	rectangle(laySplit[0],tempRect, Scalar(255),-1);// change the pixel values
			      	rectangle(isumText,tempRect, Scalar(-1),-1);
			      	rectangle(plotImg,tempRect, Scalar(255,0,0),-1);
			//cnt++;
			//cout<<"yahoooooooooooo "<<cnt<<endl;	
			}
			iter=true;
			
		}
//--------------------------------------------------------------------------------		
	integral(laySplit[1],isumText,CV_64F);
	//integral(laySplit[1],isumGraphics,CV_64F);
//---------------------------------------------------------------------------------------------------------------------------------	
	//bool 
	iter=true;
	// for testing purpose
	//int j=400,i=350;
	for (int i=0;i<imgWidth;i++)//x
		for (int j=0;j<imgHight;j++)//y
		{
			if(laySplit[0].at<uchar>(j,i)==1)
			{
				Rect tempRect;
				tempRect.x=i;tempRect.y=j;tempRect.width=50;tempRect.height=50;
				// initial density check
				double initialC=0;
				double tl= isumText.at<double>(tempRect.y,tempRect.x);
				double tr= isumText.at<double>(tempRect.y,tempRect.x+tempRect.width);
				double bl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				initialC = br-bl-tr+tl;
				if (initialC<tempRect.width*tempRect.height)
					continue;// skip if the starting pixel is not valid
				br=0;bl=0;tr=0;tl=0;
				while(iter)
				{
					// finding the width cost
					double widthCost=0;
				      	double tl= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	double tr= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width+delta));
				      	double bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	double br= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width+delta));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		widthCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	//finding height cost
				      	double heightCost=0;
				      	tl= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x+tempRect.width));
				      	bl= isumText.at<double>((tempRect.y+tempRect.height+delta),tempRect.x);
				      	br= isumText.at<double>((tempRect.y+tempRect.height+delta),(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      		heightCost = br-bl-tr+tl;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding icost
				      	double icost=0;
				      	if (tempRect.x>delta){
				      	tl= isumText.at<double>(tempRect.y,(tempRect.x-delta));
				      	tr= isumText.at<double>(tempRect.y,tempRect.x);
				      	bl= isumText.at<double>((tempRect.y+tempRect.height),(tempRect.x-delta));
				      	br= isumText.at<double>((tempRect.y+tempRect.height),tempRect.x);
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)
				      	    	icost = br-bl-tr+tl;
				      	}else icost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	// finding jcost
				      	double jcost=0;
				      	if (tempRect.y>delta){
				      	tl= isumText.at<double>((tempRect.y-delta),tempRect.x);
				      	tr= isumText.at<double>((tempRect.y-delta),tempRect.x+tempRect.width);
				      	bl= isumText.at<double>(tempRect.y,tempRect.x);
				      	br= isumText.at<double>(tempRect.y,(tempRect.x+tempRect.width));
				      	if (br>=0 and bl>=0 and tr>=0 and tl>=0)				      						      	jcost = br-bl-tr+tl;
				      	}else jcost=0;
				      	br=0;bl=0;tr=0;tl=0;
				      	if (jcost>=costParam*tempRect.width*delta)
				      		{
				      		tempRect.y=tempRect.y-delta;
				      		tempRect.height=tempRect.height+delta;
				      		}
				      	if (icost>=costParam*tempRect.height*delta)
				      		{
				      		tempRect.x=tempRect.x-delta;
				      		tempRect.width=tempRect.width+delta;
				      		}
				      	if (widthCost>=costParam*tempRect.height*delta)
				      		tempRect.width=tempRect.width+delta;
				      	if (heightCost>=costParam*tempRect.width*delta)
				      		tempRect.height=tempRect.height+delta;
				      	// condition to exit the loop
				      	if(icost<costParam*tempRect.height*delta and widthCost<costParam*tempRect.height*delta and heightCost<costParam*tempRect.width*delta and jcost<costParam*tempRect.width*delta)
				      		iter=false;
			      	}

			      	//cout<<"the rect width is = "<<tempRect.width<<endl;
			      	rectangle(laySplit[0],tempRect, Scalar(255),-1);// change the pixel values
			      	rectangle(isumText,tempRect, Scalar(-1),-1);
			      	rectangle(plotImg,tempRect, Scalar(0,255,0),-1);
			//cnt++;
			//cout<<"yahoooooooooooo "<<cnt<<endl;	
			}
			iter=true;
			
		}
//--------------------------------------------------------------------------------		




		
	return plotImg;
}


//cvpr onwords///////////////////////

//=========================main gmm parts=======================================================//
												//
												//
												//
void TrainGmmModel(string org_folder,string gt_folder,string model)
{
		ThreeMatstr finalFet;
		Mat initial(256,1,CV_32F,Scalar(0));
	//cout<<"text size = "<<initial.size()<<endl;
		finalFet.background = initial.clone();
		finalFet.figure		= initial.clone();
		finalFet.text		= initial.clone();

		vector<string> files = vector<string>();
		getdir(org_folder,files); 
//	final feture list
		Mat final_lst;
		//cout<<"testt= "<<files.size()<<endl;
		for (unsigned int i = 0;i < files.size();i++) 
		{
			// if (i==1)// looop for sample run
			// {
			// 	break;
			// }
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
        	// cout << gt_path << endl;
        	// cout << org_path << endl;
			cout<<"Now running "<<files[i]<<endl;
			Rect imageprp(p_size,0,image.cols,image.rows); 
		// making of lbp image
			Mat locbp = makeLbpImg(image);
		// calling patch listing function---
			Mat listofpatch = patchpos(imageprp);
		// creating feature and labels
			ThreeMatstr train_feture = crtTrainFet(listofpatch,locbp,gt_img);

		// concatinating the out put features
			hconcat(finalFet.text,train_feture.text,finalFet.text);
			hconcat(finalFet.figure,train_feture.figure,finalFet.figure);
			hconcat(finalFet.background,train_feture.background,finalFet.background);
		// cout<<"str ="<<finalFet.text.row(3)<<endl;
		}
	// cleaning feature data
		// ThreeMatstr clean_feture = cleanFet(finalFet);
		ThreeMatstr clean_feture = finalFet;
		int maxx = maximum(clean_feture.text.cols,clean_feture.figure.cols,clean_feture.background.cols);

		Mat textFeture = clean_feture.text.t();
		Mat figue = clean_feture.figure.t();
		Mat backGnd = clean_feture.background.t();
		// gmm model creation
		// gmm statrs/=====================================---------------------
		const int cov_mat_type = cv::EM::COV_MAT_SPHERICAL;//COV_MAT_GENERIC;//
    		cv::TermCriteria term(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 1500, 1e-3);
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
		
Mat gmmtest(Mat image,string model)
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
	
	for (int i=0;i<listofpatch.rows;i++)//
	{	

		Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		// Rect ross(listofpatch.at<float>(i,2),listofpatch.at<float>(i,1),psiz,psiz);
		//cout<<"====="<<ross<<", i= "<<i<<endl;
		patch= locbp(ross);
		// calculating histogram
		calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);
		//hist = hist/(psiz*psiz);
		double txt = gmmRealPred( gmmText, hist);
		double graP = gmmRealPred( gmmGraph, hist);
		double bac = gmmRealPred( gmmBacK, hist);

		rectangle(outImage, ross, Scalar((int)(txt),(int)(graP),(int)(bac)), -1, 8, 0 );

	}
	//imwrite("sample.png",outImage);	
	// energy minimization step
	Mat enerfyMin;
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

void gmmSave(EM gmm_text,string modName)
{
	FileStorage fsTxtSave(modName, FileStorage::WRITE);
	if (fsTxtSave.isOpened()) // if we have file with parameters, read them
	{
	    	gmm_text.write(fsTxtSave);
	    	fsTxtSave.release();
	}
}
EM readModel(string modTxt)
{
	FileStorage fsText(modTxt, FileStorage::READ);
	EM gmmText;
	if (fsText.isOpened()) // if we have file with parameters, read them
	{
		const FileNode& fnText1 = fsText["StatModel.EM"];
		gmmText.read(fnText1);
		fsText.release();
	}
	return gmmText;
}
	
double gmmRealPred(EM gmm,Mat hist)
{
	hist = hist.t();
	Mat prb;
    	Vec2d outvec = gmm.predict(hist,prb);
    	Mat weights = gmm.get<cv::Mat>("weights");
    	Mat mul = prb*weights.t();
    	double pv = mul.at<double>(0,0)*255;
	return pv;
}

Mat leptEval(Mat image)
{
	Mat gray;
	cvtColor(image, gray, CV_RGB2GRAY);
	threshold(gray,gray,125,255,THRESH_BINARY);
	Mat out(gray.size(),CV_8UC3,Scalar(0,0,0));
	for (int i=0;i<out.cols;i++)
		for (int j=0;j<out.rows;j++)
			{
				//cout<<(int)gray.at<uchar>(j,i)<<", ";
				if((int)gray.at<uchar>(j,i)==0)
				{
					out.at<Vec3b>(j,i)[0]=0;
					out.at<Vec3b>(j,i)[1]=255;
					out.at<Vec3b>(j,i)[2]=0;
				}
				else
				{
					out.at<Vec3b>(j,i)[0]=255;
					out.at<Vec3b>(j,i)[1]=0;
					out.at<Vec3b>(j,i)[2]=0;
				}				
			}
	
	return out;
}
//============= ading gabour fetures============================
Mat gabourFet(Mat image)
{
	Mat dest;
	Mat src_f;
	image.convertTo(src_f,CV_32F);
	Rect regn;
	regn.x=500;
	regn.y=1000;
	regn.width=50;
	regn.height=100;
	for (int i=0;i<1;i++) // angle is fixed
	for(int j=1;j<2;j++){
	stringstream numb;
	stringstream lamp;
	lamp<<j*10+1;
	numb<<CV_PI*i/8;
	string labda = lamp.str();
	string str = numb.str();
	string modGra ("5theta_");
	string ext (".png");
	string fulname = labda+modGra+str+ext;
	int kernel_size = 30;
	double sig = 5, th = CV_PI*i/8, lm = j*10+1, gm = 0.02, ps = 0;
	cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	clock_t tStart = clock();
	cv::filter2D(src_f, dest, CV_32F, kernel);
	Mat patch = dest(regn);
	Mat maxFet,minFet;
	reduce(patch, maxFet, 0, CV_REDUCE_SUM, CV_32F);//0-> vertical sum; 1-> Horizontal sum;
	cout<<"maxFet = "<<maxFet<<endl;
	//double min, max;
	//Scalar mean, stdev;
	//cv::minMaxLoc(dest, &min, &max);
	
	//meanStdDev(dest, mean, stdev);
	//cout<<"min val = "<<min<<endl;
	//cout<<"max val = "<<max<<endl;	
	//cout<<"mean val = "<<mean.val[0]<<endl;
	//cout<<"stdev val = "<<stdev.val[0]<<endl;	
	//dest.convertTo(dest,CV_32F,1,-min);  
	//dest.convertTo(dest,CV_32F,1/(max-min));  
	//cv::minMaxLoc(dest, &min, &max);
	//cout<<"min val = "<<min<<endl;
	//cout<<"max val = "<<max<<endl;
	printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	//----------------------------------------
	Mat viz;
	patch.convertTo(viz,CV_8U,255.0/20000);     // move to proper[0..255] range to show it
	//imshow("k",kernel);
	imwrite(fulname,viz);
	}
	//waitKey();
	return image;
}
//=========================== Gabour features ==============
Mat L2Normalization(Mat inpArray)
{
	if(feturE==4){ 
	Mat multy = inpArray*inpArray.t();
        Mat diagon =multy.diag();
        sqrt(diagon,diagon);
        Mat divident = repeat(diagon, 1,inpArray.cols);
        Mat output = inpArray/(divident+0.00001);
        return output;}
	else return inpArray;
}

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

//--------------------------------
//============***************** ctreating training features*****************////////
ThreeMatstr crtTrainFetGabur(Mat& image,Mat& gt_img, Classifier deep)
{
       // ::google::InitGoogleLogging("./docSeg");
       // string model_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/deploy.prototxt";
       // string trained_file = "/users/jobinkv/installs/caffe_cpp/googleNet/bvlc_reference_caffenet.caffemodel";
       // string mean_file    = "/users/jobinkv/installs/caffe_cpp/googleNet/imagenet_mean.binaryproto";
       // string label_file   = "/users/jobinkv/installs/caffe_cpp/googleNet/synset_words.txt";
       // Classifier deep(model_file, trained_file, mean_file, label_file);	
	// declairing feature extraction class

	fetExtrct meth1;
	// setting input image
	meth1.setInpImage(image);
	Mat gray;
	cvtColor(image,gray,CV_RGB2GRAY);
	
	
	//gabur Fetur extraction
	if (feturE==1)
	meth1.gaburFet();
	
	// lbp features
	if (feturE==2)
	meth1.lbpFet();
	
	// cc features
	if (feturE==3)
	meth1.ccFet ();	

	// deep features
	int cnnnt=0;
	//Classifier classifier();
	//if (cnnnt==0){

	//cnnnt++;

	//} 
	
	// get number of features
	int noFetr = 0;
	if (feturE<=3)
	noFetr = meth1.noOfFetr();
	else
	noFetr = 4096;
	// take the list of patches
	Mat listofpatch = meth1.listOfpatch();
	// out put feature initioalization	
	ThreeMatstr outFeture;
	Mat initial(noFetr,1,CV_32F,Scalar(0));
 	//------------------------------
	outFeture.background = initial.clone();
	outFeture.figure	= initial.clone();
	outFeture.text		= initial.clone();
	//-------------------------------
	Mat hist,label,channel[3];
	

	for (int i=0;i<listofpatch.rows;i++)
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
		}//else 
		//hist = meth1.deepFtr(ross); //deep features
//	--------creating labels for each patch-----------------------
		label = gt_img(ross);
		split(label, channel);
 
		double sum_b = cv::sum( channel[0] )[0];// text region 			=>label 0
		double sum_g = cv::sum( channel[1] )[0];// figure region		=>label 1
		double sum_r = cv::sum( channel[2] )[0];// background region	=>label 2
		
		//Assigning to curresponding lablel
		if (sum_b>=sum_g and sum_b>=sum_r)
			hconcat(outFeture.text,hist,outFeture.text);
		if (sum_g>=sum_b and sum_g>=sum_r)
			hconcat(outFeture.figure,hist,outFeture.figure);
		if (sum_r>=sum_b and sum_r>=sum_g)
			hconcat(outFeture.background,hist,outFeture.background);
		
	}
	// clear model
	meth1.clrAll();
	return outFeture;
}

//----------------------------------------gmm gabour training ends -------------------------------------
// -------------------------------------COMMON functions ---------------------------------------------
/*
Mat makeGaburImg(Mat image)
{
	image.convertTo(image,CV_32F);
	Mat out;
	int kernel_size = 30,i=2;
	double sig = 5, th = CV_PI*i/8, lm = 11, gm = 0.001, ps = 0;
	Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	filter2D(image, out, CV_32F, kernel);
	return out;
}
Mat extrGabrFet(Mat patch)
{
	Mat mins,maxs,out;
	//reduce(patch, mins, 0, CV_REDUCE_MIN, CV_32F);
	reduce(patch, maxs, 0, CV_REDUCE_MAX, CV_32F);
	//vconcat(mins,maxs,out);
	cvtColor(maxs,maxs,CV_RGB2GRAY);
	return maxs.t();
}
*/
//---------------------------------------gmm gabour testing --------------------------------------------
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
// -------------*_*_*_*_*__*_*_*_*_*_*_*_*_* feture extraction class function -----------------------
/*
void fetExtrct::setInpImage (Mat image) {

if (image.channels()==3 ){
cvtColor(image,grayImg,CV_RGB2GRAY);
  	inpImg = grayImg.clone();
  	}
  	else
  	inpImg = image.clone();
  	
}
*/
void fetExtrct::setInpImage (Mat image) {
  	inpImg = image.clone();
  	if (inpImg.channels()==3 )
  	cvtColor(inpImg,grayImg,CV_RGB2GRAY);
  	else
  	inpImg = grayImg;
  	
}


void fetExtrct::gaburFet () {
 	inpImg.convertTo(inpImg,CV_32F);
 	Mat downSmp;
 	resize(inpImg, downSmp, Size(),(double)1/2, (double)1/2, INTER_NEAREST);
	int kernel_size = 12,i=5;
	double sig = 6, th = CV_PI*i/8, lm = 11, gm = 0.001, ps = 0;
	Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	filter2D(downSmp, fetImage1, CV_32F, kernel); 
	//th = CV_PI;
	//Mat kernel2 = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	//filter2D(downSmp, fetImage2, CV_32F, kernel2); 
	//fetImage1.convertTo(fetImage1,CV_8UC1);
	//Mat view=fetImage1.clone();
	//normalize(view, view, 0, 1, CV_MINMAX);
	resize(fetImage1, fetImage1, inpImg.size(), INTER_NEAREST);
	//namedWindow( "Display window", WINDOW_NORMAL );// Create a window for display.
	//imshow( "Display window", view ); 
	//waitKey(0); 	
	//resize(fetImage2, fetImage2, Size(),(double)2, (double)2, INTER_NEAREST);
	noFtr = p_size;

}

/*

void fetExtrct::gaburFet () {
 	inpImg.convertTo(inpImg,CV_32F);
 	Mat downSmp;
 	resize(inpImg, downSmp, Size(),(double)1/2, (double)1/2, INTER_NEAREST);
	int kernel_size = 30,i=2;
	double sig = 5, th = CV_PI*i/8, lm = 11, gm = 0.001, ps = 0;
	Mat kernel = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	filter2D(downSmp, fetImage1, CV_32F, kernel); 
	//th = CV_PI;
	//Mat kernel2 = cv::getGaborKernel(cv::Size(kernel_size,kernel_size), sig, th, lm, gm, ps);
	//filter2D(downSmp, fetImage2, CV_32F, kernel2); 
	resize(fetImage1, fetImage1, inpImg.size(), INTER_NEAREST);
	//resize(fetImage2, fetImage2, Size(),(double)2, (double)2, INTER_NEAREST);
	noFtr = p_size;

}


*/

void fetExtrct::lbpFet () {
	
	int radius = 1;
	int neighbors = 8;
	Mat dst; 
// pre-processing part
    	cvtColor(inpImg, dst, CV_BGR2GRAY);
	GaussianBlur(dst, dst, Size(7,7), 5, 3, BORDER_CONSTANT); // tiny bit of smoothing is always a good idea
    // -----------------------different lbp functions-----------------------------------
    //lbp::ELBP(dst, locbp, radius, neighbors); // use the extended operator
	lbp::OLBP(dst, locbp); // use the original operator
    	//lbp::VARLBP(dst, locbp, radius, neighbors);
    // a simple min-max norm will do the job...
	normalize(locbp, locbp, 0, 255, NORM_MINMAX, CV_32FC1);
	noFtr = 256;

}
void fetExtrct::ccFet () {
	Mat areasOpenImg = areaOpen(inpImg, areaRem);
	dobleMat ccfetr = labeledFeatures(areasOpenImg);
	LabelImg = ccfetr.labeledImg;
	labelFeature = ccfetr.featureMat;
	noFtr = 6;
}

Mat fetExtrct::ccftrXtr(Rect ross)
{
	Mat ccpatch = LabelImg(ross).clone();
	int mod = findModmat( ccpatch);
	Mat patchFet = getPatchFet(labelFeature, mod);
	return patchFet;
}


Mat fetExtrct::deepFtr(Rect ross)
{
	Mat ccpatch = inpImg(ross).clone();//TODO//
	int mod = findModmat( ccpatch);
	Mat patchFet = getPatchFet(labelFeature, mod);
	return patchFet;
}

Mat fetExtrct::lbpftr(Rect ross)
{
	int histSize = 256;
	float range[] = { 0, 256 } ; //the upper boundary is exclusive
	const float* histRange = { range };
	Mat hist;
	bool uniform = true; bool accumulate = false;
	Mat patch;
	
	patch= locbp(ross);
		// calculating histogram
	calcHist( &patch, 1, 0, Mat(),hist, 1, &histSize, &histRange, uniform, accumulate);

	return hist;

	
}

int fetExtrct::noOfFetr () {
  	return noFtr;

}

Mat fetExtrct::features(Rect ross)
{
	Mat patch1 = fetImage1(ross),maxs1,maxs2,fett;
	//patch2 = fetImage2(ross),
	//=---------------------
	reduce(patch1, maxs1, 0, CV_REDUCE_MAX, CV_32F);
	//reduce(patch2, maxs2, 1, CV_REDUCE_MAX, CV_32F);
	//vconcat(mins,maxs,out);
	cvtColor(maxs1,maxs1,CV_RGB2GRAY);
	//cvtColor(maxs2,maxs2,CV_RGB2GRAY);
	//hconcat(maxs1,maxs2.t(),fett);
	return maxs1.t();
}

Mat fetExtrct::listOfpatch() // list ofpatches
{
	int nobk=0;
	nobk = floor( ((double) inpImg.rows/p_size))*floor( ((double) inpImg.cols/p_size)) +floor( ((double) inpImg.rows/p_size))+floor( ((double) inpImg.cols/p_size))+1;
	
	Mat codi_list(nobk,3, CV_32F,Scalar(0));
	int cnt=0;
	for (int i =0;i<inpImg.rows-1-p_size;i=i+p_size)//Row
	{
		for (int j =0;j<inpImg.cols-1-p_size;j=j+p_size)//Col
		{
			codi_list.at<float>(cnt,0)=cnt;
			codi_list.at<float>(cnt,1)=i;
			codi_list.at<float>(cnt,2)=j;
			cnt=cnt+1;
		}
	}
	//cout<<"1st "<<cnt<<endl;
	for (int i =0;i<inpImg.rows-1-p_size;i=i+p_size)//Row
	{

		int j=inpImg.cols-2-p_size; // -2 because the lbp function gives out image 2 row/col less than the original
			codi_list.at<float>(cnt,0)=cnt;
		codi_list.at<float>(cnt,1)=i;
		codi_list.at<float>(cnt,2)=j;
		cnt=cnt+1;
	}	
	//cout<<"2nd "<<cnt<<endl;
	for (int j =0;j<inpImg.cols-1-p_size;j=j+p_size)//Row
	{

		int i=inpImg.rows-2-p_size;
		codi_list.at<float>(cnt,0)=cnt;
		codi_list.at<float>(cnt,1)=i;
		codi_list.at<float>(cnt,2)=j;
		cnt=cnt+1;

	}

	//cout<<"3rd "<<cnt<<endl;
	codi_list.at<float>(cnt,0)=cnt;
	codi_list.at<float>(cnt,1)=inpImg.rows-p_size-2;
	codi_list.at<float>(cnt,2)=inpImg.cols-p_size-2;
	//cout<<"4th "<<cnt<<endl;
	return codi_list;

}
void fetExtrct::clrAll()
{
	inpImg.release();
	fetImage1.release();
	fetImage2.release();
	fetr.release();

}
//-------------------------------------------------------
ThreeMatstr list2mat1(vector<ThreeMatstr> feturelist)
{
	ThreeMatstr finalFet;
	int fetdim = feturelist[0].background.rows;
	Mat initial(fetdim,1,CV_32F,Scalar(0));
	//cout<<"text size = "<<initial.size()<<endl;
	finalFet.background 	= initial.clone();
	finalFet.figure		= initial.clone();
	finalFet.text		= initial.clone();
	for(int i=0; i<feturelist.size();i++)
	{
		// concatinating the out put features
		hconcat(finalFet.text,feturelist[i].text,finalFet.text);
		hconcat(finalFet.figure,feturelist[i].figure,finalFet.figure);
		hconcat(finalFet.background,feturelist[i].background,finalFet.background);		
	}
	
	return finalFet;
}
//--------------------------------cc features ---------------------------------
Mat areaOpen(Mat binary, int area)
{

	threshold(binary,binary, 0, 1, CV_THRESH_OTSU);
    // Fill the labelImage with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
	
	cv::Mat labelImage, outImage(binary.rows,binary.cols,CV_8UC1,Scalar(0));
	binary.convertTo(labelImage,  CV_32SC1);
	int labelCount = 2; // starts at 2 because 0,1 are used already
	for(int y=0; y < labelImage.rows; y++)
	        for(int x=0; x < labelImage.cols; x++) 
	        {
            		if(labelImage.at<int>(y,x)!= 1)
                		continue;
		        cv::Rect rect;
		        cv::floodFill(labelImage, cv::Point(x,y), labelCount, &rect, 0, 0, 8);
		        std::vector <cv::Point2i> blob;
		        for(int i=rect.y; i < (rect.y+rect.height); i++) 
                		for(int j=rect.x; j < (rect.x+rect.width); j++) 
                		{
                    			if(labelImage.at<int>(i,j) != labelCount) 
                        			continue;
                   			blob.push_back(cv::Point2i(j,i));
                		}
            		if (blob.size()>=area)// removing the small areas
            			for (int i=0;i<blob.size();i++)
            				outImage.at<uchar>(blob[i].y,blob[i].x)=255;

            		labelCount++;
        	}

    	return outImage;
}

//--------------------------------simple text detection  ---------------------------------
int textdet(Mat binary, int area)
{

	threshold(binary,binary, 0, 1, CV_THRESH_OTSU);
    // Fill the labelImage with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
	int info=0;
	cv::Mat labelImage, outImage(binary.rows,binary.cols,CV_8UC1,Scalar(0));
	binary.convertTo(labelImage,  CV_32SC1);
	int labelCount = 2; // starts at 2 because 0,1 are used already
	for(int y=0; y < labelImage.rows; y++)
	        for(int x=0; x < labelImage.cols; x++) 
	        {
            		if(labelImage.at<int>(y,x)!= 1)
                		continue;
		        cv::Rect rect;
		        cv::floodFill(labelImage, cv::Point(x,y), labelCount, &rect, 0, 0, 8);
		        std::vector <cv::Point2i> blob;

            		if (rect.width>=area or rect.height>=area)// checking width and height
				{info = 1;
				return info;
				break;
				}

            		labelCount++;
        	}

    	return info;
}

//--------------------------------------------------------------
dobleMat labeledFeatures(Mat binary)
{

	dobleMat outPut;
	threshold(binary,binary, 0, 1, CV_THRESH_OTSU);
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground
	
	cv::Mat labelImage, outImage(binary.rows,binary.cols,CV_8UC1,Scalar(0));
	binary.convertTo(labelImage,  CV_32SC1);
	int fetLen = 7;
//	features //
	cv::Mat features(1,(fetLen),CV_32F,Scalar(0));
	cv::Mat features1(1,(fetLen),CV_32F,Scalar(1));
	vconcat(features, features1, features);
	//cout<<features<<endl;
	int labelCount = 2; // starts at 2 because 0,1 are used already
	for(int y=0; y < labelImage.rows; y++)
	        for(int x=0; x < labelImage.cols; x++) 
	        {
	        	
            		if(labelImage.at<int>(y,x)!= 1)
                		continue;
		        cv::Rect rect;
		        cv::floodFill(labelImage, cv::Point(x,y), labelCount, &rect, 0, 0, 8);
		        Mat temp(rect.height,rect.width,CV_8UC1,Scalar(0));
//		        std::vector <cv::Point2i> blob;
		        int ccArea=0;
		        for(int i=rect.y; i < (rect.y+rect.height); i++) 
                		for(int j=rect.x; j < (rect.x+rect.width); j++) 
                		{
                    			if(labelImage.at<int>(i,j) != labelCount) 
                        			continue;
                   			temp.at<uchar>(i-rect.y,j-rect.x) = 255;
                   			ccArea++;
                		}
                	if(labelCount!=1 )// binary feature extraction//
                	{
                		//stringstream ss1;
				//ss1 << labelCount<<"patch.png";
				//string str1 = ss1.str();
				//imwrite(str1, temp);
				cv::Mat featurest(1,fetLen,CV_32F,Scalar(0));
				//find_moments(temp);
				Moments mu;
				mu = moments( temp, true );
				//cout<<"Moment ="<<mu.m00<<endl;
				//cout<<"ccArea ="<<ccArea<<endl;
				
				double hu[7]={0,0,0,0,0,0,0};
				HuMoments(mu,hu);
				//cout<<"pach no ="<<labelCount<<endl;
				for(int jj=0;jj<7;jj++)
					featurest.at<float>(0,jj)=hu[jj];
				//featurest.at<float>(0,fetLen)=labelCount;
				//vconcat(features, featurest, features);
				// calculating hog
				//HOGDescriptor hog( Size(128,128), Size(128,128), Size(32,32), Size(32,32), 9); ///144 features
				//vector<float> ders;
				//vector<Point>locs;
				// 
				//resize(temp, temp, Size(128,128), INTER_NEAREST);
				//hog.compute(temp,ders,Size(0,0),Size(0,0),locs);
				//Mat Hogfeat(1,ders.size(),CV_32FC1,Scalar(0));
				//for(int kk=7;kk<(7+144);kk++)
					//featurest.at<float>(0,kk)=ders.at(kk-7);

				vconcat(features, featurest, features);
				//imwrite("temp.png",temp);
				//cout<<"Wrote"<<featurest.size()<<endl;
				//cout<<"features ="<<features.size()<<endl;
				//cout<<"features ="<<features<<endl;
				
				
				
			}
			//cout<<"ccArea ="<<ccArea<<endl;
			//cout<<labelCount<<endl;
            		labelCount++;
            		
        	}
//cout<<"features ="<<features<<endl;
outPut.labeledImg = labelImage; 
outPut.featureMat = features;
    	return outPut;
}

//---------------------------------------------------------------
int findModmat(Mat ccpatch)
{

	double minVal; 
	double maxLabel; 
	Point minLoc; 
	Point maxLoc;

	minMaxLoc( ccpatch, &minVal, &maxLabel, &minLoc, &maxLoc );
	//std::vector<int> counts(maxLabel, 0);
		////////////////////////////////////////////
	cv::Mat linearcc = ccpatch.reshape ( 0, 1 );
	
		//cout<<"rows = "<<linearcc.cols<<endl;
	Mat location;
	if (maxLabel>=1)
	{
		location =  cv::Mat::zeros ( 1,(maxLabel+1), CV_32S );
		for (int a = 0; a < linearcc.cols; a++){
			int loc = linearcc.at<int>(0, a);
				if (loc!=0)
		 			location.at<int>(0,loc) += 1;}
		    	//counts.at(2)+=1;
			//cout<<"location ="<<location.sum()<<endl;
			//cout<<location<<endl;
		double minVal; 
		double maxLabel; 
		Point minLoc; 
		Point maxLoc;

		minMaxLoc( location, &minVal, &maxLabel, &minLoc, &maxLoc );
		//cout<<"maxLoc"<<maxLoc.x<<endl;
		return maxLoc.x;
	}
	else return 0;
}
//--------------------------------------------------------
Mat getPatchFet(Mat labelFeature, int label)
{
	Mat out(labelFeature.cols-1,1,CV_32F,Scalar(0));
	for(int i=0; i<labelFeature.cols-1;i++)
		out.at<float>(i,0)=labelFeature.at<float>(label,i);
		//cout<<labelFeatureat<float>(label,i)<<endl;;
	return out;
}

