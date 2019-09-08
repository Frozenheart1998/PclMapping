#include <iostream>
#include <fstream>
using namespace std;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry> 
#include <boost/format.hpp>  // for formating strings
#include <pcl/point_types.h> 
#include <pcl/io/pcd_io.h> 
#include <pcl/visualization/pcl_visualizer.h>
#include "3rdParty/pfmLib/ImageIOpfm.h"
#include <math.h>
#include <gflags/gflags.h>
#include "3rdParty/libelas/elas.h"
#include "3rdParty/libelas/image.h"

DEFINE_double(max_depth,
			  7000,
			  "The maximum depth in one frame.");
DEFINE_int32(frame_num,
			  100,
			  "The total number of processing frames.");
DEFINE_int32(start_frame,
			  0,
			  "The starting frame index.");
DEFINE_int32(maxdisp,
			 255,
			 "The maximum of disparity.");
DEFINE_bool(enable_elas,
			 false,
			 "Enable libelas.");
#define PI	3.1415926

// compute disparities of pgm image input pair grayLeftImgs, grayRightImgs
void process (cv::Mat left, cv::Mat right, string outfile, int no_interp)
{
	clock_t c0 = clock();

	// load images
	image<uchar> *I1,*I2;
	I1 = new image<uchar>(left.cols, left.rows, true);
	I2 = new image<uchar>(right.cols, right.rows, true);

	memcpy(I1->data, left.data, left.cols * left.rows * sizeof(uchar));
	memcpy(I2->data, right.data, right.cols * right.rows * sizeof(uchar));
	/*for(int y=0; y<left.rows; y++){
		for(int x=0; x<left.cols; x++){
			uchar* left_row_ptr = left.ptr<uchar> (y);
			uchar* right_row_ptr = right.ptr<uchar> (y);
			I1->data[y+x*left.rows] = left_row_ptr[x];
			I2->data[y+x*left.rows] = right_row_ptr[x];
		}
	}*/

	// check for correct size
	if (I1->width()<=0 || I1->height() <=0 || I2->width()<=0 || I2->height() <=0 ||
		I1->width()!=I2->width() || I1->height()!=I2->height()) {
		cout << "ERROR: Images must be of same size, but" << endl;
		cout << "       I1: " << I1->width() <<  " x " << I1->height() <<
			 ", I2: " << I2->width() <<  " x " << I2->height() << endl;
		delete I1;
		delete I2;
		return;
	}

	// get image width and height
	int32_t width  = I1->width();
	int32_t height = I1->height();

	// allocate memory for disparity images
	const int32_t dims[3] = {width,height,width}; // bytes per line = width
	float* D1_data = (float*)malloc(width*height*sizeof(float));
	float* D2_data = (float*)malloc(width*height*sizeof(float));

	// process
	Elas::parameters param(Elas::MIDDLEBURY);
	if (no_interp) {
		//param = Elas::parameters(Elas::ROBOTICS);
		// don't use full 'robotics' setting, just the parameter to fill gaps
		param.ipol_gap_width = 3;
	}
	param.postprocess_only_left = false;
	param.disp_max = FLAGS_maxdisp;
	Elas elas(param);
	elas.process(I1->data,I2->data,D1_data,D2_data,dims);

	// added runtime output - DS 4/4/2013
	clock_t c1 = clock();
	double secs = (double)(c1 - c0) / CLOCKS_PER_SEC;
	printf("runtime: %.2fs  (%.2fs/MP)\n", secs, secs/(width*height/1000000.0));

	// save disparity image
	cv::Mat disp(height, width, CV_32FC1, (void*)D1_data);
	//Mat (int rows, int cols, int type, void *data, size_t step=AUTO_STEP)
	WriteFilePFM(disp, outfile, 1.0f/FLAGS_maxdisp);

	// free memory
	delete I1;
	delete I2;
	free(D1_data);
	free(D2_data);
}

int main( int argc, char** argv )
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
    vector<cv::Mat> colorImgs, inspImgs; // color, inspection images
    cv::Mat grayLeftImg, grayRightImg; // instant gray image
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses; // camera poses

    ifstream fin("../trajectory.txt");
    if (!fin)
    {
        cerr<<"trajectory.txt excluded!"<<endl;
        return 1;
    }

    if(FLAGS_enable_elas){
		for ( int i=FLAGS_start_frame; i<100; i++ ){
			boost::format fmt( "../Img/%s/%s%d.%s" ); // image file format
			if(i<10){
				grayLeftImg  = cv::imread( (fmt%"gray_left"%"00000"%i%"png").str() );
				grayRightImg = cv::imread( (fmt%"gray_right"%"00000"%i%"png").str() );
				//ELAS process
				process(grayLeftImg, grayRightImg,(fmt%"depth"%"00000"%i%"pfm").str(), 0);
			}
			else{
				grayLeftImg  = cv::imread( (fmt%"gray_left"%"0000"%i%"png").str() );
				grayRightImg = cv::imread( (fmt%"gray_right"%"0000"%i%"png").str() );
				//ELAS process
				process(grayLeftImg, grayRightImg,(fmt%"depth"%"0000"%i%"pfm").str(), 0);
			}
		}
    }
    
    for ( int i=FLAGS_start_frame; i<100; i++ )
    {
    	cv::Mat out;
        boost::format fmt( "../Img/%s/%s%d.%s" ); // image file format
        if(i<10){
        	colorImgs.push_back( cv::imread( (fmt%"color_left"%"00000"%i%"png").str() ));
			ReadFilePFM(out, (fmt%"depth"%"00000"%i%"pfm").str());
        	inspImgs.push_back( out );
        }
        else{
        	colorImgs.push_back( cv::imread( (fmt%"color_left"%"0000"%i%"png").str() ));
			ReadFilePFM(out, (fmt%"depth"%"0000"%i%"pfm").str());
			inspImgs.push_back( out );
        }

		double data[12] = {0};
        for ( auto& d:data ){
			fin>>d;
        }

        Eigen::Matrix3d rotMatrix;
        rotMatrix << data[0], data[1], data[2],
        			 data[4], data[5], data[6],
        			 data[8], data[9], data[10];
        Eigen::Isometry3d T( rotMatrix );
        T.pretranslate( Eigen::Vector3d( data[3], data[7], data[11] ));
        poses.push_back( T );



    }

    //x axis rotate 7*pi/12
	Eigen::Matrix3d adjustMatrix;
	adjustMatrix << 1, 0, 0,
					0, cos(7*PI/12), -sin(7*PI/12),
					0, sin(7*PI/12), cos(7*PI/12);
	Eigen::Isometry3d adjustT( adjustMatrix );
    
    // calculate pointclouds
    // camera parameters
    double cx = 707.0912;
    double cy = 707.0912;
    double fx = 601.8873;
    double fy = 183.1104;
    double bf = 379.8145; // baseline
    double depthScale = 1.0;
    
    cout<<"converting images into pointclouds"<<endl;
    
    // defines pcl type
    typedef pcl::PointXYZRGB PointT; 
    typedef pcl::PointCloud<PointT> PointCloud;
    
    // init new pointcloud
    PointCloud::Ptr pointCloud( new PointCloud ); 
    for ( int i=0; i<FLAGS_frame_num; i++ )
    {
        cout<<"converting images... "<<i+1<<endl;
        cv::Mat color = colorImgs[i];
        cv::Mat insp = inspImgs[i];
        Eigen::Isometry3d T = poses[i];
		cout << "T=\n" << T.matrix() << endl;
		for ( int v=0; v<color.rows; v++ )
            for ( int u=0; u<color.cols; u++ )
            {
                double inspection = insp.ptr<float>(v)[u]; //\brief get inspection value
                if ( inspection==0 ) continue; // o represents that inspection value undetected
                Eigen::Vector3d point;
				point[2] = bf * fx / inspection; // depth = baseline * f / inspection
				if( point[2] > FLAGS_max_depth ) point[2] = 0;
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;
                //Eigen::Vector3d pointWorld = adjustT*T*point;
				Eigen::Vector3d pointWorld = T*point;
                
                PointT p ;
                p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];
                p.b = color.data[ v*color.step+u*color.channels() ];
                p.g = color.data[ v*color.step+u*color.channels()+1 ];
                p.r = color.data[ v*color.step+u*color.channels()+2 ];
                pointCloud->points.push_back( p );
            }
    }
    
    pointCloud->is_dense = false;
    cout<<"Totall points in pointcloud: "<<pointCloud->size()<<endl;
    pcl::io::savePCDFileBinary("map.pcd", *pointCloud );
    return 0;
}
