#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;
Mat L2Normalization(Mat inpArray)
{
        Mat multy = inpArray*inpArray.t();
        Mat diagon =multy.diag();
        sqrt(diagon,diagon);
        Mat divident = repeat(diagon, 1,inpArray.cols);
        Mat output = inpArray/(divident+0.00001);
        return output;
}

int main( int argc, char** argv )
{
        Mat A = (Mat_<float>(2, 3) << 1.5, 2.3, 3.8, 4.2, 5.1, 6.7);
        cout<<"A = "<<A<<endl;
        cout<<"A.size() = "<<A.size()<<endl;
        cout<<"A.t() = "<<A.t()<<endl;
        cout<<"A.t().size() = "<<A.t().size()<<endl;
        cout<<"A*A.t() = "<<A*A.t()<<endl;
        Mat B = L2Normalization(A);
        cout<<"B = "<<B<<endl;
        //cout<<B.size()<<endl;

        return 0;
}
