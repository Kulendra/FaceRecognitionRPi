#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>

//new
#include </usr/local/include/opencv/cv.h>
#include </usr/local/include/opencv/highgui.h>
#include "time.h"

extern "C" {
	#include "/opt/vc/userland-master/host_applications/linux/libs/bcm_host/include/bcm_host.h"
	#include "/opt/vc/userland-master/interface/vcos/vcos.h"
	
	#include "/opt/vc/userland-master/interface/mmal/mmal.h"
	#include "/opt/vc/userland-master/interface/mmal/mmal_logging.h"
	#include "/opt/vc/userland-master/interface/mmal/mmal_buffer.h"
	#include "/opt/vc/userland-master/interface/mmal/util/mmal_util.h"
	#include "/opt/vc/userland-master/interface/mmal/util/mmal_util_params.h"
	#include "/opt/vc/userland-master/interface/mmal/util/mmal_default_components.h"
	#include "/opt/vc/userland-master/interface/mmal/util/mmal_connection.h"
	
	#include "RaspiCamControl.h"
	#include "RaspiPreview.h"
	#include "RaspiCLI.h"
}

#include <semaphore.h>


// OPENCV
#include </usr/include/c++/4.6/iostream>
#include </usr/include/c++/4.6/fstream>
#include </usr/include/c++/4.6/sstream>
#include "time.h"

#include "/usr/local/include/opencv2/core/core.hpp"
#include "/usr/local/include/opencv2/contrib/contrib.hpp"
#include "/usr/local/include/opencv2/highgui/highgui.hpp"
#include "/usr/local/include/opencv2/imgproc/imgproc.hpp"
#include "/usr/local/include/opencv2/objdetect/objdetect.hpp"

using namespace cv;
using namespace std;

// some constants to manage nb of people to learn+ id of people 
#define MAX_PEOPLE 		3

#define DEVENDRA		1

#define KULDEEP			2

#define OTHER			0

// for debug and trace
#define TRACE 1
#define DEBUG_MODE 0
#define DEBUG if (DEBUG_MODE==1)

CascadeClassifier face_cascade; 
CvPoint Myeye_left;
CvPoint Myeye_right;
Ptr<FaceRecognizer> model = createFisherFaceRecognizer() ;
string fn_haar;
string fn_csv;
int im_width;		// image width
int im_height;		// image height
int PREDICTION_SEUIL ;
char key;
int filenumber = 250;
string filename; ////////////////////////////////////////////////////////////	

Mat gray,frame,face,face_resized;
vector<Mat> images;
vector<int> labels;

// name of people
string  people[MAX_PEOPLE];

// nb of times RPI talks to one guy
int nbSpeak[MAX_PEOPLE];

int bHisto;
//vector< Rect_<int> > faces;

// nb of picture learnt by people
int nPictureById[MAX_PEOPLE];
///////////////////////

/// Camera number to use - we only have one camera, indexed from 0.
#define CAMERA_NUMBER 0

// Standard port setting for the camera component
#define MMAL_CAMERA_PREVIEW_PORT 0
#define MMAL_CAMERA_VIDEO_PORT 1
#define MMAL_CAMERA_CAPTURE_PORT 2

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Video format information
#define VIDEO_FRAME_RATE_NUM 30
#define VIDEO_FRAME_RATE_DEN 1

/// Video render needs at least 2 buffers.
#define VIDEO_OUTPUT_BUFFERS_NUM 3

// Max bitrate we allow for recording
const int MAX_BITRATE = 30000000; // 30Mbits/s


// variable to convert I420 frame to IplImage
int nCount=0;
IplImage *py, *pu, *pv, *pu_big, *pv_big, *image,* dstImage;
		
		
int mmal_status_to_int(MMAL_STATUS_T status);

/** Structure containing all state information for the current run
 */
typedef struct
{
   int timeout;                        /// Time taken before frame is grabbed and app then shuts down. Units are milliseconds
   int width;                          /// Requested width of image
   int height;                         /// requested height of image
   int bitrate;                        /// Requested bitrate
   int framerate;                      /// Requested frame rate (fps)
   int graymode;			/// capture in gray only (2x faster)
   int immutableInput;      /// Flag to specify whether encoder works in place or creates a new buffer. Result is preview can display either
                                       /// the camera output or the encoder output (with compression artifacts)
   RASPIPREVIEW_PARAMETERS preview_parameters;   /// Preview setup parameters
   RASPICAM_CAMERA_PARAMETERS camera_parameters; /// Camera setup parameters

   MMAL_COMPONENT_T *camera_component;    /// Pointer to the camera component
   MMAL_COMPONENT_T *encoder_component;   /// Pointer to the encoder component
   MMAL_CONNECTION_T *preview_connection; /// Pointer to the connection from camera to preview
   MMAL_CONNECTION_T *encoder_connection; /// Pointer to the connection from camera to encoder

   MMAL_POOL_T *video_pool; /// Pointer to the pool of buffers used by encoder output port
   
} RASPIVID_STATE;

/** Struct used to pass information in encoder port userdata to callback
 */
typedef struct
{
   FILE *file_handle;                   /// File handle to write buffer data to.
   VCOS_SEMAPHORE_T complete_semaphore; /// semaphore which is posted when we reach end of frame (indicates end of capture or fault)
   RASPIVID_STATE *pstate;            /// pointer to our state in case required in callback
} PORT_USERDATA;

////////////////////////////////////////
/////////////////////////////////////////////////
// trace if TRACE==1
/////////////////////////////////////////////////
void trace(string s)
{
	if (TRACE==1)
	{
		cout<<s<<"\n";
	}
}
/////////////////////////////////////////////////
//
// read csv files

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "(E) No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    int nLine=0;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) 
        {
        	// read the file and build the picture collection
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
            nPictureById[atoi(classlabel.c_str())]++;
        	nLine++;
        }
    }
    
    // write number of picture by people
    // did you notice? I'm not familiar with string classe :-)
    // I prefer my old good char* ...
    // (what a pity)
	char sTmp[128];
    sprintf(sTmp,"(init) %d pictures read to train",nLine);
    trace((string)(sTmp));
	for (int j=0;j<MAX_PEOPLE;j++)
	{
		sprintf(sTmp,"(init) %d pictures of %s (%d) read to train",nPictureById[j],people[j].c_str(),j);
   	 	trace((string)(sTmp));
	}
}


/////////////////////////////////////////

// default status
static void default_status(RASPIVID_STATE *state)
{
   if (!state)
   {
      vcos_assert(0);
      return;
   }

   // Default everything to zero
   memset(state, 0, sizeof(RASPIVID_STATE));

   // Now set anything non-zero
   state->timeout 			= 120000;     // capture time : here 65 s
   state->width 			= 320;      // use a multiple of 320 (640, 1280)
   state->height 			= 240;		// use a multiple of 240 (480, 960)
   state->bitrate 			= 17000000; // This is a decent default bitrate for 1080p
   state->framerate 		= VIDEO_FRAME_RATE_NUM;
   state->immutableInput 	= 1;
   state->graymode 			= 1;		//gray by default, much faster than color (0), mandatory for face reco
   
   // Setup preview window defaults
   raspipreview_set_defaults(&state->preview_parameters);

   // Set up the camera_parameters to default
   raspicamcontrol_set_defaults(&state->camera_parameters);
}



/**
 *  buffer header callback function for video
 *
 * @param port Pointer to port from which callback originated
 * @param buffer mmal buffer header pointer
 */
static void video_buffer_callback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer)
{
   MMAL_BUFFER_HEADER_T *new_buffer;
   PORT_USERDATA *pData = (PORT_USERDATA *)port->userdata;

   if (pData)
   {
     
      if (buffer->length)
      {

	      mmal_buffer_header_mem_lock(buffer);
 
 		//
		// *** PR : OPEN CV Stuff here !
		//
		int w=pData->pstate->width;	// get image size
		int h=pData->pstate->height;
		int h4=h/4;
		
		memcpy(py->imageData,buffer->data,w*h);	// read Y
		
		if (pData->pstate->graymode==0)
		{
			memcpy(pu->imageData,buffer->data+w*h,w*h4); // read U
			memcpy(pv->imageData,buffer->data+w*h+w*h4,w*h4); // read v
	
			cvResize(pu, pu_big, CV_INTER_NN);
			cvResize(pv, pv_big, CV_INTER_NN);  //CV_INTER_LINEAR looks better but it's slower
			cvMerge(py, pu_big, pv_big, NULL, image);
	
			cvCvtColor(image,dstImage,CV_YCrCb2RGB);	// convert in RGB color space (slow)
			gray=cvarrToMat(dstImage);   
			cvShowImage("camcvWin", dstImage );//////////////////////////////////////////////////////////////////////////000000//////////////////////////
			
		}
		else
		{	
			// for face reco, we just keep gray channel, py
			gray=cvarrToMat(py);  
			cvShowImage("camcvWin", py); // display only gray channel///////////////////////////////////////////////00000////////////////////////////////
		}
		
////////////////////////////////
// FACE RECOGNITION START HERE
////////////////////////////////

	std::vector<Rect> faces;

	// detect faces
	face_cascade.detectMultiScale(gray, faces, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30,30));
	// for each faces founded

// Set Region of Interest
    	cv::Rect roi_b;
    	cv::Rect roi_c;

    	size_t i = 0; // ic is index of current element
    	int ac = 0; // ac is area of current element

    	size_t ib = 0; // ib is index of biggest element
    	int ab = 0; // ab is area of biggest element

	for(int i = 0; i < faces.size(); i++) 
	{       

		roi_c.x = faces[i].x;
	        roi_c.y = faces[i].y;
	        roi_c.width = (faces[i].width);
	        roi_c.height = (faces[i].height);
	
	        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)
	
	        roi_b.x = faces[ib].x;
	        roi_b.y = faces[ib].y;
	        roi_b.width = (faces[ib].width);
	        roi_b.height = (faces[ib].height);
	
	        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element
	
	        if (ac > ab)
	        {
	            ib = i;
	            roi_b.x = faces[ib].x;
	            roi_b.y = faces[ib].y;
	            roi_b.width = (faces[ib].width);
	            roi_b.height = (faces[ib].height);
        	}	

		// crop face (pretty easy with opencv, don't you think ? 
		Rect face_i = roi_b;
		
		face = gray(face_i);  
		//  resized face and display it
		cv::resize(face, face_resized, Size(90,90 ), 1.0, 1.0, CV_INTER_NN); // This will be needed later while saving images
		//cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, CV_INTER_NN); //INTER_CUBIC);	

		// Form a filename
        	filename = "";
        	stringstream ssfn;
        	ssfn << "/home/pi/Visitors/"<< filenumber << ".png";
        	filename = ssfn.str();
        	filenumber++;

        	imwrite(filename, face_resized);	
	
		// now, we try to predict who is it ? 
		char sTmp[256];		
		double predicted_confidence	= 0.0;
		int prediction				= -1;
		prediction = model->predict(face_resized);
		
		// create a rectangle around the face      
		rectangle(gray, roi_b, CV_RGB(255, 255 ,255), 1);
	
		// if good prediction : > threshold 
		//if (predicted_confidence>PREDICTION_SEUIL)
		//{
			// trace
			//sprintf(sTmp,"+ prediction ok = %s (%d) confiance = (%d)",people[prediction].c_str(),prediction,(int)predicted_confidence);
			trace((string)(sTmp));
		
		 	// display name of the guy on the picture
			string box_text; /////////////////////////////////////////////////////////////////////////////////////////00000000000///////////////////////////////
			if (prediction<MAX_PEOPLE)
			{
				box_text = "Id="+people[prediction];
			}
			else
			{
				trace("(E) prediction id incohÃƒÂ©rent");
			}
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);			   
			putText(gray, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,255,255), 1.0);	
		        		
		//}
		//else
		//{		
				// trace is commented to speed up
				//sprintf(sTmp,"- prediction too low = %s (%d) confiance = (%d)",people[prediction].c_str(),prediction,(int)predicted_confidence);
				//trace((string)(sTmp));
	//}
	} // end for
	
			
/////////////////////////
// END OF FACE RECO
/////////////////////////
		
	// Show the result:
	imshow("camcvWin", gray); /////////////////////////////////////////////////////////////////////0000000000000000000///////////////////////////////////////////////
	key = (char) waitKey(1);
	printf("%d\n",nCount);
	nCount++;		// count frames displayed 
		
         mmal_buffer_header_mem_unlock(buffer);
      }
      else vcos_log_error("buffer null");
      
   }
   else
   {
      vcos_log_error("Received a encoder buffer callback with no state");
   }

   // release buffer back to the pool
   mmal_buffer_header_release(buffer);

   // and send one back to the port (if still open)
   if (port->is_enabled)
   {
      MMAL_STATUS_T status;

      new_buffer = mmal_queue_get(pData->pstate->video_pool->queue);

      if (new_buffer)
         status = mmal_port_send_buffer(port, new_buffer);

      if (!new_buffer || status != MMAL_SUCCESS)
         vcos_log_error("Unable to return a buffer to the encoder port");
   }
    
}


/**
 * Create the camera component, set up its ports
 *
 * @param state Pointer to state control struct
 *
 * @return 0 if failed, pointer to component if successful
 *
 */
static MMAL_COMPONENT_T *create_camera_component(RASPIVID_STATE *state)
{
	MMAL_COMPONENT_T *camera = 0;
	MMAL_ES_FORMAT_T *format;
	MMAL_PORT_T *preview_port = NULL, *video_port = NULL, *still_port = NULL;
	MMAL_STATUS_T status;
	
	/* Create the component */
	status = mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &camera);
	
	if (status != MMAL_SUCCESS)
	{
	   vcos_log_error("Failed to create camera component");
	   goto error;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	if (!camera->output_num)
	{
	   vcos_log_error("Camera doesn't have output ports");
	   goto error;
	}
	
	video_port = camera->output[MMAL_CAMERA_VIDEO_PORT];
	still_port = camera->output[MMAL_CAMERA_CAPTURE_PORT];
	
	//  set up the camera configuration
	{
	   MMAL_PARAMETER_CAMERA_CONFIG_T cam_config =
	   {
	      { MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config) },
	      cam_config.max_stills_w = state->width,
	      cam_config.max_stills_h = state->height,
	      cam_config.stills_yuv422 = 0,
	      cam_config.one_shot_stills = 0,
	      cam_config.max_preview_video_w = state->width,
	      cam_config.max_preview_video_h = state->height,
	      cam_config.num_preview_video_frames = 3,
	      cam_config.stills_capture_circular_buffer_height = 0,
	      cam_config.fast_preview_resume = 0,
	      cam_config.use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC
	   };
	   mmal_port_parameter_set(camera->control, &cam_config.hdr);
	}
	// Set the encode format on the video  port
	
	format = video_port->format;
	format->encoding_variant = MMAL_ENCODING_I420;
	format->encoding = MMAL_ENCODING_I420;
	format->es->video.width = state->width;
	format->es->video.height = state->height;
	format->es->video.crop.x = 0;
	format->es->video.crop.y = 0;
	format->es->video.crop.width = state->width;
	format->es->video.crop.height = state->height;
	format->es->video.frame_rate.num = state->framerate;
	format->es->video.frame_rate.den = VIDEO_FRAME_RATE_DEN;
	
	status = mmal_port_format_commit(video_port);
	if (status)
	{
	   vcos_log_error("camera video format couldn't be set");
	   goto error;
	}
	
	// PR : plug the callback to the video port 
	status = mmal_port_enable(video_port, video_buffer_callback);
	if (status)
	{
	   vcos_log_error("camera video callback2 error");
	   goto error;
	}

   // Ensure there are enough buffers to avoid dropping frames
   if (video_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
      video_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;


   // Set the encode format on the still  port
   format = still_port->format;
   format->encoding = MMAL_ENCODING_OPAQUE;
   format->encoding_variant = MMAL_ENCODING_I420;
   format->es->video.width = state->width;
   format->es->video.height = state->height;
   format->es->video.crop.x = 0;
   format->es->video.crop.y = 0;
   format->es->video.crop.width = state->width;
   format->es->video.crop.height = state->height;
   format->es->video.frame_rate.num = 1;
   format->es->video.frame_rate.den = 1;

   status = mmal_port_format_commit(still_port);
   if (status)
   {
      vcos_log_error("camera still format couldn't be set");
      goto error;
   }

	
	//PR : create pool of message on video port
	MMAL_POOL_T *pool;
	video_port->buffer_size = video_port->buffer_size_recommended;
	video_port->buffer_num = video_port->buffer_num_recommended;
	pool = mmal_port_pool_create(video_port, video_port->buffer_num, video_port->buffer_size);
	if (!pool)
	{
	   vcos_log_error("Failed to create buffer header pool for video output port");
	}
	state->video_pool = pool;

	/* Ensure there are enough buffers to avoid dropping frames */
	if (still_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
	   still_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;
	
	/* Enable component */
	status = mmal_component_enable(camera);
	
	if (status)
	{
	   vcos_log_error("camera component couldn't be enabled");
	   goto error;
	}
	
	raspicamcontrol_set_all_parameters(camera, &state->camera_parameters);
	
	state->camera_component = camera;
	
	return camera;

error:

   if (camera)
      mmal_component_destroy(camera);

   return 0;
}

/**
 * Destroy the camera component
 *
 * @param state Pointer to state control struct
 *
 */
static void destroy_camera_component(RASPIVID_STATE *state)
{
   if (state->camera_component)
   {
      mmal_component_destroy(state->camera_component);
      state->camera_component = NULL;
   }
}


/**
 * Destroy the encoder component
 *
 * @param state Pointer to state control struct
 *
 */
static void destroy_encoder_component(RASPIVID_STATE *state)
{
   // Get rid of any port buffers first
   if (state->video_pool)
   {
      mmal_port_pool_destroy(state->encoder_component->output[0], state->video_pool);
   }

   if (state->encoder_component)
   {
      mmal_component_destroy(state->encoder_component);
      state->encoder_component = NULL;
   }
}

/**
 * Connect two specific ports together
 *
 * @param output_port Pointer the output port
 * @param input_port Pointer the input port
 * @param Pointer to a mmal connection pointer, reassigned if function successful
 * @return Returns a MMAL_STATUS_T giving result of operation
 *
 */
static MMAL_STATUS_T connect_ports(MMAL_PORT_T *output_port, MMAL_PORT_T *input_port, MMAL_CONNECTION_T **connection)
{
   MMAL_STATUS_T status;

   status =  mmal_connection_create(connection, output_port, input_port, MMAL_CONNECTION_FLAG_TUNNELLING | MMAL_CONNECTION_FLAG_ALLOCATION_ON_INPUT);

   if (status == MMAL_SUCCESS)
   {
      status =  mmal_connection_enable(*connection);
      if (status != MMAL_SUCCESS)
         mmal_connection_destroy(*connection);
   }

   return status;
}

/**
 * Checks if specified port is valid and enabled, then disables it
 *
 * @param port  Pointer the port
 *
 */
static void check_disable_port(MMAL_PORT_T *port)
{
   if (port && port->is_enabled)
      mmal_port_disable(port);
}

/**
 * Handler for sigint signals
 *
 * @param signal_number ID of incoming signal.
 *
 */
static void signal_handler(int signal_number)
{
   // Going to abort on all signals
   vcos_log_error("Aborting program\n");

   // TODO : Need to close any open stuff...how?

   exit(255);
}

/**
 * main
 */
int main(int argc, const char **argv)
{
	
	
/////////////////////////////////
// BEGIN OF FACE RECO INIT
/////////////////////////////////

	//
	// see thinkrpi.wordpress.com, articles on Magic Mirror to understand this command line and parameters
	//
	cout<<"start\n";
	   if ((argc != 1)) {
	       printf("Hey! Just use ./camcv");
	       exit(1);
	   }
	
	// set value by default for prediction treshold = minimum value to recognize
	if (argc==1) { trace("(init) prediction threshold = 4500.0 by default");PREDICTION_SEUIL = 4500.0;}
	//if (argc==4) PREDICTION_SEUIL = atoi(argv[3]);
	PREDICTION_SEUIL = 4500.0 ;
	
	// do we do a color histogram equalization ? 
	bHisto= 1;
	

	// init people, should be do in a config file,
	// but I don't have time, I need to go to swimming pool
	// with my daughters
	// and they prefer to swimm than to see their father do a config file
	// life is hard.
	people[OTHER] 	= "UNKNOWN";
	people[DEVENDRA] 	= "Devendra";
	//people[JOHNCENA] 	= "Johncena";
	people[KULDEEP]		= "Kuldeep";
	
	// init...
	// reset counter
	for (int i=0;i>MAX_PEOPLE;i++) 
	{
		nPictureById[i]=0;
	}
	int bFirstDisplay	=1;
	trace("(init) People initialized");
	
	// Get the path to your CSV
	fn_csv = string("/home/pi/camcv/database.txt");
	
	// Note : /!\ change with your opencv path	
	//fn_haar = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
	// change fn_harr to be quicker LBP (see article)
	fn_haar = "/home/pi/Downloads/opencv-2.4.7/data/haarcascades/haarcascade_frontalface_default.xml";
	DEBUG cout<<"(OK) csv="<<fn_csv<<"\n";
	
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_csv, images, labels);
		DEBUG cout<<"(OK) read CSV ok\n";
    	} 
    catch (cv::Exception& e) 
    {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }

	// get heigh, witdh of 1st images--> must be the same
    im_width = images[0].cols;
    im_height = images[0].rows;
	trace("(init) Images are OK!");
 
 	//
    // Create a FaceRecognizer and train it on the given images:
	//
	
	// this a Eigen model, but you could replace with Fisher model (in this case
	// threshold value should be lower) (try)	
	
    //	Fisherfaces model; 
    
    // train the model with your nice collection of pictures	
    trace("(init) start train images");
    model->train(images, labels);
 	trace("(init) Images are Trained.");
 
	// load face model
    if (!face_cascade.load(fn_haar))
   	{
    			cout <<"(E) face cascade model not loaded :"+fn_haar+"\n"; 
    			return -1;
    }
    trace("(init) Face cascade model loaded.");
    
/////////////////////////////////
// END OF FACE RECO INIT
/////////////////////////////////
	
	
	// Our main data storage vessel..
	RASPIVID_STATE state;
	
	MMAL_STATUS_T status;// = -1;
	MMAL_PORT_T *camera_video_port = NULL;
	MMAL_PORT_T *camera_still_port = NULL;
	MMAL_PORT_T *preview_input_port = NULL;
	MMAL_PORT_T *encoder_input_port = NULL;
	MMAL_PORT_T *encoder_output_port = NULL;
	
	time_t timer_begin,timer_end;
	double secondsElapsed;
	
	bcm_host_init();
	signal(SIGINT, signal_handler);

	// read default status
	default_status(&state);

	// init windows and OpenCV Stuff
	cvNamedWindow("camcvWin", CV_WINDOW_AUTOSIZE); 
	int w=state.width;
	int h=state.height;
	dstImage = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 3);
	py = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);		// Y component of YUV I420 frame
	pu = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);	// U component of YUV I420 frame
	pv = cvCreateImage(cvSize(w/2,h/2), IPL_DEPTH_8U, 1);	// V component of YUV I420 frame
	pu_big = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
	pv_big = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 1);
	image = cvCreateImage(cvSize(w,h), IPL_DEPTH_8U, 3);	// final picture to display

   
	// create camera
	if (!create_camera_component(&state))
	{
	   vcos_log_error("%s: Failed to create camera component", __func__);
	}
	else if ((status = raspipreview_create(&state.preview_parameters)) != MMAL_SUCCESS)
	{
	   vcos_log_error("%s: Failed to create preview component", __func__);
	   destroy_camera_component(&state);
	}
	else
	{
		PORT_USERDATA callback_data;
		
		camera_video_port   = state.camera_component->output[MMAL_CAMERA_VIDEO_PORT];
		camera_still_port   = state.camera_component->output[MMAL_CAMERA_CAPTURE_PORT];
	   
		VCOS_STATUS_T vcos_status;
		
		callback_data.pstate = &state;
		
		vcos_status = vcos_semaphore_create(&callback_data.complete_semaphore, "RaspiStill-sem", 0);
		vcos_assert(vcos_status == VCOS_SUCCESS);
		
		// assign data to use for callback
		camera_video_port->userdata = (struct MMAL_PORT_USERDATA_T *)&callback_data;
        
        // init timer
  		time(&timer_begin); 

       
       // start capture
		if (mmal_port_parameter_set_boolean(camera_video_port, MMAL_PARAMETER_CAPTURE, 1) != MMAL_SUCCESS)
		{
		   	return 0;
		}
		
		// Send all the buffers to the video port
		
		int num = mmal_queue_length(state.video_pool->queue);
		int q;
		for (q=0;q<num;q++)
		{
		   MMAL_BUFFER_HEADER_T *buffer = mmal_queue_get(state.video_pool->queue);
		
		   if (!buffer)
		   		vcos_log_error("Unable to get a required buffer %d from pool queue", q);
		
			if (mmal_port_send_buffer(camera_video_port, buffer)!= MMAL_SUCCESS)
		    	vcos_log_error("Unable to send a buffer to encoder output port (%d)", q);
		}
		
		
		// Now wait until we need to stop
		vcos_sleep(state.timeout);
  
		//mmal_status_to_int(status);
		// Disable all our ports that are not handled by connections
		check_disable_port(camera_still_port);
		
		if (state.camera_component)
		   mmal_component_disable(state.camera_component);
		
		//destroy_encoder_component(&state);
		raspipreview_destroy(&state.preview_parameters);
		destroy_camera_component(&state);
		
		}
	if (status != 0)
	raspicamcontrol_check_configuration(128);
	
	time(&timer_end);  /* get current time; same as: timer = time(NULL)  */
	cvReleaseImage(&dstImage);
	cvReleaseImage(&pu);
	cvReleaseImage(&pv);
	cvReleaseImage(&py);
	cvReleaseImage(&pu_big);
	cvReleaseImage(&pv_big);
	
	secondsElapsed = difftime(timer_end,timer_begin);
	
	printf ("%.f seconds for %d frames : FPS = %f\n", secondsElapsed,nCount,(float)((float)(nCount)/secondsElapsed));
		
   return 0;
}
