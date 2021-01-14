#include "text_recognition.h"
#include "utils.h"
//#include "opencv2/opencv.hpp"


using namespace std;
using namespace caffe;
using namespace cv;



static vector<string>   g_extendednames;
static vector<string>   share_images;
static vector<string>   share_labels;
static unsigned int share_image_index = 0;
static float accuracy = 0;
static int curr_correct_character_num = 0;
static int curr_stati_character_num = 0;

static inline string& trim(string &s, const string& mask)
{
	if (s.empty())
	{
		return s;
	}

	s.erase(0, s.find_first_not_of(mask));
	s.erase(s.find_last_not_of(mask) + 1);
	return s;
}

static int read_txt_list(const string& filename, vector<string>& list, int index)
{
	ifstream fin(filename);
	string line;
	while (getline(fin, line))
	{
		line = trim(line, " ");
		vector<string> subStrings = splitString(line, " ", true);
		string split_txt = subStrings[index];
		//DEBUG_PRINT(<< split_txt  << endl);
		list.push_back(split_txt);
	}
	return 0;
}



int main(int argc, char **argv)
{
	const std::string deviceName = "gpu:0,1";

	const std::string rec_label_file = "../../1s2n.list";
	const std::string rec_model_file = "../../model-jiandao/deploy-stnpt-fnn-vgg-pro-stnfeature-64x256.prototxt";
	const std::string rec_weights_file = "../../model-jiandao/stnpt-fnn-vgg-pro-stnfeature-xyxy2032_iter_600000.caffemodel";


	CTextRecognitionCache* recognition_ptr = TextRecognitionCacheSingleton::get_instance(rec_model_file, rec_weights_file, deviceName, rec_label_file, 0.0, 0.0, 2);

	if (argc < 2){
		DEBUG_PRINT(<< "the argc is less than 2." << endl);
		return -1;
	}

	//Caffe::set_mode(Caffe::CPU);

	string img_path = "../../test-images2/42_1.jpg";

	// string img_path = argv[1];
	string out_path = "../tmp/";
	if (argc == 3)
		string out_path = argv[2];


	vector<string> out_text;
	int detector_index = recognition_ptr->RequestDetector();
	if (detector_index < 0)
	{
		DEBUG_PRINT(<< "logo_getDetector failed. " << endl);
		return -1;
	}
	CTextRecognition* working_detect = (CTextRecognition*)recognition_ptr->GetDetector(detector_index);
	if (working_detect)
	{
		cv::Mat curr_image = cv::imread(img_path);

		vector<string> recognized_strings;
		//Mat converted_img = ConvertMode(dst_image, 3);
		working_detect->RecognizeSTN(curr_image, recognized_strings, 1);
		out_text = recognized_strings;

		recognition_ptr->FreeDetector(detector_index);
	}

	vector<string>& ref_strings = out_text;
	string ss_res = "";
	for (int j = 0; j < ref_strings.size(); j++) {
		string uft_string;
		uft_string = Utf8ToGbk(ref_strings[j]);
		//uft_string = ref_strings[j];
		ss_res += uft_string + "$$$$";
	}
	cout << "catch it "  << ": " << ss_res << endl;


	return 0;
}