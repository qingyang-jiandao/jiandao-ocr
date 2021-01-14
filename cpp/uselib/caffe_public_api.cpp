#include "caffe_public_api.h"
#include "text_recognition.h"
#include "utils.h"
#include <string.h>

//#ifdef OPENCV
#if 1

#include <opencv2/opencv.hpp>            // C++
#include <opencv2/core/version.hpp>
#ifndef CV_VERSION_EPOCH     // OpenCV 3.x and 4.x
#include <opencv2/videoio/videoio.hpp>
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#else     // OpenCV 2.x
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#ifndef USE_CMAKE_LIBS
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_video" OPENCV_VERSION ".lib")
#endif    // USE_CMAKE_LIBS
#endif    // CV_VERSION_EPOCH

#endif    // OPENCV

static void calloc_error()
{
    fprintf(stderr, "Calloc error\n");
    exit(EXIT_FAILURE);
}

static void *xcalloc(size_t nmemb, size_t size) {
    void *ptr = calloc(nmemb, size);
    if (!ptr) {
        calloc_error();
    }
    return ptr;
}

void free_detection_boxes(detbox *dets, int n)
{
    int i;
    for (i = 0; i < n; ++i) {
    }
    free(dets);
}

LIB_API void free_ptr(void *ptr, int n)
{
    if(ptr)free(ptr);
}


LIB_API void caffe_recognize_image(char *cfgfile, char *weightfile, char *label_file, char *input_filename, char *output_text, int *num)
{
	const std::string deviceName = "gpu:0";
	const std::string rec_label_file = "../model/text_rec/dict_6862_ex_ex.list";
	const std::string rec_model_file = "../model/text_rec/deploy-stn_pt-fnn-lossweight-vgg-64x256-t48.prototxt";
	//const std::string rec_weights_file = "../model/text_rec/stnpt_fnn_vgg_lw0.9_64x256_t48_iter_720000.caffemodel";
	//const std::string rec_weights_file = "../model/text_rec/train-stn_pt-fnn-vgg-64x256-t48_iter_810000.caffemodel";
	const std::string rec_weights_file = "../model/text_rec/train-stn_pt-fnn-vgg-64x256-t48-newdata_iter_1500000.caffemodel";
	//std::shared_ptr<CTextRecognitionCache> recognition_ptr = TextRecognitionCacheSingleton::get_instance(rec_model_file, rec_weights_file, deviceName, rec_label_file, 0.8, 0.3, 1);


	CTextRecognitionCache* recognition_ptr = TextRecognitionCacheSingleton::get_instance(cfgfile, weightfile, deviceName, label_file, 0.8, 0.3, 1);

	int detector_index = recognition_ptr->RequestDetector();
	if (detector_index < 0)
	{
		DEBUG_PRINT(<< "logo_getDetector failed. " << endl);
		return;
	}
	CTextRecognition* working_detect = (CTextRecognition*)recognition_ptr->GetDetector(detector_index);
	if (working_detect)
	{
		cv::Mat curr_image = cv::imread(input_filename);
		if (curr_image.empty() || !curr_image.data)
		{
			DEBUG_PRINT(<< "unable open image:" << input_filename << endl);
			return;
		}

		{
			vector<string> recognized_strings;
			//Mat converted_img = ConvertMode(dst_image, 3);
			working_detect->Recognize(curr_image, recognized_strings, 1);
			if (recognized_strings.size() > 0)
			{
				//string gbk_string;
				//gbk_string = Utf8ToGbk(recognized_strings[0]);

				string uft_string = recognized_strings[0];

				//DEBUG_PRINT(<< "recog text:" << uft_string << endl);
			}
			/*strcpy_s(output_text, recognized_strings[0].length(), recognized_strings[0].c_str());*/
			strcpy_s(output_text, strlen(recognized_strings[0].c_str())+1, recognized_strings[0].c_str());
			*num = recognized_strings[0].length();
		}
		recognition_ptr->FreeDetector(detector_index);
	}
}

LIB_API void network_recognition_init(char *cfgfile, char *weightfile, char *label_file, char *device_name, int max_model_num)
{
	CTextRecognitionCache* recognition_ptr = TextRecognitionCacheSingleton::get_instance(cfgfile, weightfile, device_name, label_file, 0.8, 0.3, max_model_num);
}

//LIB_API void network_recognize_image(char *cfgfile, char *weightfile, char *label_file, image_t im, char *output_text, int *num)
LIB_API void network_recognize_image(image_t im, char *output_text, int *num)
{
	CTextRecognitionCache* recognition_ptr = TextRecognitionCacheSingleton::get_instance();

	if (im.h <= 0 || im.w <= 0 || im.data == 0 || recognition_ptr == nullptr)
	{
		DEBUG_PRINT(<< "c++ recognize image info height " << im.h << " , width " << im.w << " data ptr " << long long(im.data) << endl);
		return;
	}

	int detector_index = recognition_ptr->RequestDetector();
	if (detector_index < 0)
	{
		DEBUG_PRINT(<< "request recognitor failed. " << endl);
		return;
	}
	CTextRecognition* working_detect = (CTextRecognition*)recognition_ptr->GetDetector(detector_index);
	if (working_detect)
	{
		//cv::Mat curr_image = cv::imread(input_filename);
		//if (curr_image.empty() || !curr_image.data)
		//{
		//	DEBUG_PRINT(<< "unable open image:" << input_filename << endl);
		//	return;
		//}

		cv::Mat curr_image(im.h, im.w, CV_32FC3, im.data);

		{
			vector<string> recognized_strings;
			//Mat converted_img = ConvertMode(dst_image, 3);
			working_detect->RecognizeSTN(curr_image, recognized_strings, 1);
			if (recognized_strings.size() > 0)
			{
				//string gbk_string;
				//gbk_string = Utf8ToGbk(recognized_strings[0]);

				string uft_string = recognized_strings[0];

				//DEBUG_PRINT(<< "recog text:" << uft_string << endl);
			}
			/*strcpy_s(output_text, recognized_strings[0].length(), recognized_strings[0].c_str());*/
			strcpy_s(output_text, strlen(recognized_strings[0].c_str()) + 1, recognized_strings[0].c_str());
			*num = recognized_strings[0].length();
		}
		recognition_ptr->FreeDetector(detector_index);
	}
}

LIB_API void network_recognize_image_n(image_t im, char **output_text, int *lengths, int N)
{
	CTextRecognitionCache* recognition_ptr = TextRecognitionCacheSingleton::get_instance();

	if (im.h <= 0 || im.w <= 0 || im.data == 0 || recognition_ptr == nullptr)
	{
		DEBUG_PRINT(<< "c++ recognize image info height " << im.h << " , width " << im.w << " data ptr " << long long (im.data) << endl);
		return;
	}

	int detector_index = recognition_ptr->RequestDetector();
	if (detector_index < 0)
	{
		DEBUG_PRINT(<< "request recognitor failed. " << endl);
		return;
	}
	CTextRecognition* working_detect = (CTextRecognition*)recognition_ptr->GetDetector(detector_index);
	if (working_detect)
	{
		//cv::Mat curr_image = cv::imread(input_filename);
		//if (curr_image.empty() || !curr_image.data)
		//{
		//	DEBUG_PRINT(<< "unable open image:" << input_filename << endl);
		//	return;
		//}

		cv::Mat curr_image(im.h, im.w, CV_32FC3, im.data);

		{
			vector<string> recognized_strings;
			//Mat converted_img = ConvertMode(dst_image, 3);
			working_detect->RecognizeSTN(curr_image, recognized_strings, N);
			if (recognized_strings.size() > 0)
			{
				for(int i=0; i< recognized_strings.size(); i++)
				{
					//string gbk_string;
					//gbk_string = Utf8ToGbk(recognized_strings[0]);

					string uft_string = recognized_strings[i];

					//DEBUG_PRINT(<< "recog text:" << uft_string << endl);

					strcpy_s(output_text[i], strlen(recognized_strings[i].c_str()) + 1, recognized_strings[i].c_str());
					lengths[i] = recognized_strings[i].length();
				}
			}
		}
		recognition_ptr->FreeDetector(detector_index);
	}
}

LIB_API void network_init()
{   
	static bool init_flag = false;
	if (!init_flag)
	{
		google::InitGoogleLogging("XXX");
		google::SetCommandLineOption("GLOG_minloglevel", "2");
		init_flag = true;
	}
}

LIB_API void network_release()
{
	CTextRecognitionCache*& recognition_ptr = TextRecognitionCacheSingleton::get_instance();
	if (recognition_ptr)delete recognition_ptr;
	recognition_ptr = nullptr;
}