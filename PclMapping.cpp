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
#include "./pfmLib/ImageIOpfm.h"

int main( int argc, char** argv )
{
    vector<cv::Mat> colorImgs, inspImgs; //color and inspection images
    vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses; // camera poses
    
    ifstream fin("../trajectory.txt");
    if (!fin)
    {
        cerr<<"trajectory.txt excluded!"<<endl;
        return 1;
    }
    
    for ( int i=0; i<100; i++ )
    {
    	cv::Mat out;
        boost::format fmt( "../%s/%s%d.%s" ); // image file format
        if(i<10){
        	colorImgs.push_back( cv::imread( (fmt%"color"%"00000"%i%"png").str() ));
			ReadFilePFM(out, (fmt%"depth"%"00000"%i%"pfm").str());
        	inspImgs.push_back( out );
        }
        else{
        	colorImgs.push_back( cv::imread( (fmt%"color"%"0000"%i%"png").str() ));
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
    for ( int i=0; i<10; i++ )
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
                point[0] = (u-cx)*point[2]/fx;
                point[1] = (v-cy)*point[2]/fy;
                Eigen::Vector3d pointWorld = T*point;
                
                PointT p ;
                /*p.x = pointWorld[0];
                p.y = pointWorld[1];
                p.z = pointWorld[2];*/
				p.x = pointWorld[0];
				p.y = -pointWorld[2];
				p.z = pointWorld[1];
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
