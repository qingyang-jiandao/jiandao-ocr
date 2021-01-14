#include "common.h"
#include "utils.h"
#include "text_recognition.h"

using namespace std;
using namespace cv;
using namespace caffe;


CTextRecognition::CTextRecognition(const string& model_file, const string& weights_file, const string& deviceName, const string& label_file)
    :DetectorBase(model_file, weights_file, deviceName)
{
	int gpu_id = -1;
	string delimStr = "gpu:";
	string::size_type pos = deviceName.find(delimStr, 0);
	if (pos != string::npos) {
		string str_gpu_id = deviceName.substr(pos + delimStr.length(), deviceName.length());
		gpu_id = atoi(str_gpu_id.c_str());
	}

	if (gpu_id<0)
	{
		Caffe::set_mode(Caffe::CPU);
	}
	else
	{
		Caffe::set_mode(Caffe::GPU);
		Caffe::SetDevice(gpu_id);
	}
}

CTextRecognition::~CTextRecognition()
{
}

bool CTextRecognition::LoadModel(const string& model_file, const string& trained_file)
{
	// Load the network.

	string::size_type pos = model_file.find(".prototxt");
	string output_bin = model_file;
	if (pos != string::npos)
	{
		NetParameter param;
		ReadNetParamsFromTextFileOrDie(model_file, &param);
		string delimStr = ".prototxt";
		output_bin = model_file.substr(0, pos) + ".bin";
		WriteProtoToBinaryFile(param, output_bin);
	}

	NetParameter param;
	const int level = 0; const vector<string>* stages = NULL;
	ReadProtoFromBinaryFile(output_bin, &param);
	// Set phase, stages and level
	param.mutable_state()->set_phase(TEST);
	if (stages != NULL) {
		for (int i = 0; i < stages->size(); i++) {
			param.mutable_state()->add_stage((*stages)[i]);
		}
	}
	param.mutable_state()->set_level(level);
	net_.reset(new Net<float>(param));

    net_->CopyTrainedLayersFrom(trained_file);


    // Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = 3;
    /*
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    */
    if (num_channels_ != 3 && num_channels_ != 1) {
		DEBUG_PRINT (<< "Input layer should have 1 or 3 channels." << endl);
        return false;
    }

    return true;

}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(std::vector<float> iv, int N) {
    int ind;
    float max;
    std::vector<int> result;
    std::vector<float> v;
    v = iv;
    if (iv.size() < N) {
        N = iv.size();
    }

    for(int i = 0; i < N; i++) {
         ind = 0;
         max = -1e10;
         for(size_t j = 0; j < v.size(); j++) {
              if(v[j] > max) {
                  max = v[j];
                  ind = j;
              }
         }
         v[ind] = -1e10;
         result.push_back(ind);
    }

    return result;
}


#if 0
// Return the indices of the top N values of vector v.
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
#endif

#if 0
// Return the top N predictions.
string CTextRecognition::Detect(const cv::Mat& img, int N) {

    clock_t start1,t_cl;
    start1 = clock();
    std::vector<float> output = Predict(img);
    t_cl = clock();
    printf("Foward Time Spent:%f\n",1.0*(t_cl-start1)/CLOCKS_PER_SEC);

    std::vector<float> temp;
    std::vector<int> maxid;
    std::vector<int> maxpathid;
    std::vector<string> prevpred;
    std::vector<string> newpred;
    std::vector<float> prevscore;

    for(int k=0;k<21019;k++) {
        temp.push_back(output[k]);
    }

    maxid = Argmax(temp,N);
    for(int i=0;i<N;i++) {
        prevscore.push_back(output[maxid[i]]);
        prevpred.push_back(std::to_string(maxid[i]));
    }

    std::vector<int> curlabel;
    std::vector<float> curscore;
    std::vector<float> pathscore;
    string finalpreds("");
    std::vector<string> preduck(N);

    for(int i=21019;i<output.size();i=i+21019) {

        temp.resize(0);
        curlabel.resize(0);
        curscore.resize(0);
        pathscore.resize(0);
        newpred.resize(0);
        for (int j=0;j<21019;j++) {
            temp.push_back(output[i+j]);
        }

        maxid = Argmax(temp,N);
        for(int k=0;k<N;k++) {
            curscore.push_back(temp[maxid[k]]);
            curlabel.push_back(maxid[k]);
        }

        for (int m=0;m<N;m++){
            for(int n=0;n<N;n++) {
                pathscore.push_back(prevscore[m]+curscore[n]);
            }
        }

        maxpathid = Argmax(pathscore,N);
        for(int q=0;q<N;q++){
            newpred.push_back(prevpred[maxpathid[q]/N]+" "+std::to_string(curlabel[maxpathid[q]%N]));
        }

        for(int q=0;q<N;q++){
            prevpred[q]=newpred[q];
            prevscore[q]=pathscore[maxpathid[q]];
        }
    }

    for(int i=0;i<N;i++){
        std::istringstream f(newpred[i]);
        string s;
        string prev = "21018";
        string space = "21018";
        int x;
        while (getline(f, s, ' ')) {
            if(s!=prev&&s!=space) {
                stringstream(s)>>x;
                preduck[i] = preduck[i] + labels_[x];
            }
            prev = s;
        }
    }

    std::vector<float> finalscore(N);
    for(int i=0;i<N;i++) {
            finalscore[i] = prevscore[i];
            for(int j=i+1;j<N;j++){
            if (preduck[i]==preduck[j]) {
                preduck[j] = "";
                finalscore[i] = finalscore[i] + prevscore[j];
            }
        }
    }

    std::vector<int> rank = Argmax(finalscore,N);
    for (int i=0;i<rank.size();i++) {
        if (preduck[rank[i]]!="") {
            finalpreds = finalpreds + preduck[rank[i]] + "\n";
        }
    }

    return finalpreds;
}
#endif

static void ConvertBlobToMat(caffe::shared_ptr<Blob<float>> blob, std::vector<float>& v_float, cv::Mat& outimg) {
	int c = blob->channels();
	int h = blob->height();
	int w = blob->width();
	int n = blob->num();

	if (blob->shape().size() == 1) {
		v_float.resize(n);
		for (int i = 0; i < n; i++) {
			v_float[i] = blob->data_at(i, 0, 0, 0);
		}
	}
	if (blob->shape().size() == 2) {
		outimg = cv::Mat::zeros(n, c, CV_32FC1);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < c; j++) {
				outimg.at<float>(i, j) = blob->data_at(i, j, 0, 0);
			}
		}
		outimg.convertTo(outimg, CV_8UC1);
	}

	if (blob->shape().size() == 4) {
		vector<Mat> tmpmats;
		for (int j = 0; j < c; j++) {
			cv::Mat tmp_mat(h, w, CV_32FC1, (void*)((float*)blob->cpu_data() + (j * h * w)));
			tmpmats.push_back(tmp_mat);
		}
		if (c == 3 || c == 4)
		{
			Mat merged_img;
			merge(tmpmats, merged_img);
			cv::normalize(merged_img, outimg, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		}
		else if (c == 1)
		{
			cv::normalize(tmpmats[0], outimg, 0, 255, cv::NORM_MINMAX, CV_8UC3);
		}
	}
}

static void getThetas(caffe::shared_ptr<Blob<float>> blob, std::vector<float>& v_float) {
	int c = blob->channels();
	int h = blob->height();
	int w = blob->width();
	int n = blob->num();

	{
		v_float.resize(6);
		for (int i = 0; i < c; i++) {
			v_float[i] = blob->data_at(0, i, 0, 0);
		}
	}
}

void CTextRecognition::Transform(const cv::Mat& img, cv::Mat& loc_img, const string& input_name, const string& output_name)
{
	boost::shared_ptr<Blob<float>> input_layer = net_->blob_by_name(input_name);
	int height = img.size().height;
	int width = img.size().width;
	//int height_new = input_geometry_.height;
	int height_new = 32;
	int width_new = int(1.0*height_new*width / height);
	if (width_new < 128)
	{
		width_new = 128;
	}

	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	// processing imgs
	cv::Mat sample_resized;
	cv::Size input_geometry_new = cv::Size(width_new, height_new);
	cv::resize(sample, sample_resized, input_geometry_new);
	if (width_new > 128)
	{
		width_new = 128;
		sample_resized = sample_resized(cv::Range(0, height_new), cv::Range(0, width_new));
	}
	//cv::imwrite("../tmp/init_img.jpg", sample_resized);

	input_layer->Reshape(1, num_channels_, height_new, width_new);
	//DEBUG_PRINT(<< "num_channels " << input_layer->channels() << " height_new " << input_layer->height() << " width_new " << input_layer->width() << endl);
	//net_->Reshape();

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_iv;
	//cv::divide(sample_float, 255, sample_iv, 1);
	sample_iv = sample_float*(1.0 / 255);

	float* input_data = input_layer->mutable_cpu_data();
	vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layer->channels(); ++i)
	{
		cv::Mat channel(height_new, width_new, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += height_new * width_new;
	}

	cv::split(sample_iv, input_channels);

	//net_->Forward();
	///* Copy the output layer to a std::vector */
	//boost::shared_ptr<caffe::Blob<float> > outlayer = net_->blob_by_name(output_name);
	////int small_width = outlayer->width();
	////int small_height = outlayer->height();
	////int small_channels = outlayer->channels();
	////DEBUG_PRINT(<< " small_channels " << small_channels << " small_height " << small_height << " small_width " << small_width << endl );

	//std::vector<float> v_float;
	//getThetas(outlayer, v_float);

	//cv::Point2f srcTri[3];
	//cv::Point2f dstTri[3];
	//srcTri[0] = Point2f(0, 0);
	//srcTri[1] = Point2f(width_new - 1, 0);
	//srcTri[2] = Point2f(0, height_new - 1);

	//dstTri[0] = Point2f(v_float[3] * -1 + v_float[4] * -1 + v_float[5], v_float[0] * -1 + v_float[1] * -1 + v_float[2]);
	//dstTri[1] = Point2f(v_float[3] * -1 + v_float[4] * 1 + v_float[5], v_float[0] * -1 + v_float[1] * 1 + v_float[2]);
	//dstTri[2] = Point2f(v_float[3] * 1 + v_float[4] * -1 + v_float[5], v_float[0] * 1 + v_float[1] * -1 + v_float[2]);

	//dstTri[0] = Point2f((dstTri[0].x + 1) / 2 * width_new, (dstTri[0].y + 1) / 2 * height_new);
	//dstTri[1] = Point2f((dstTri[1].x + 1) / 2 * width_new, (dstTri[1].y + 1) / 2 * height_new);
	//dstTri[2] = Point2f((dstTri[2].x + 1) / 2 * width_new, (dstTri[2].y + 1) / 2 * height_new);

	//cv::Mat warp_mat = getAffineTransform(srcTri, dstTri);
	////cv::Mat warp_mat(2, 3, CV_32FC1, (float*)v_float.data());

	//int target_height = input_geometry_.height;
	//int target_width = int(1.0*target_height*width / height);
	//Mat warp_dstImage = Mat::zeros(target_height, target_width, img.type());
	//cv::Size target_geometry = cv::Size(target_width, target_height);
	//cv::resize(sample, sample_resized, target_geometry);
	//warpAffine(sample_resized, warp_dstImage, warp_mat, warp_dstImage.size());

	//static int count = 0;
	//count++;
	//{
	//	string newName = "../tmp/loc_img_" + to_string(count) + ".jpg";
	//	cv::imwrite(newName, warp_dstImage);
	//	newName = "../tmp/sample_resized_" + to_string(count) + ".jpg";
	//	cv::imwrite(newName, sample_resized);
	//}

	//loc_img = warp_dstImage;
}

//void PrintMat(Mat& A)
//{
//	for (int i = 0; i < A.rows; i++)
//	{
//		for (int j = 0; j < A.cols; j++)
//			cout << A.at<float>(i, j) << ' ';
//		cout << endl;
//	}
//	cout << endl;
//}

/* Wrap the input layer of the network in separate cv::Mat objects
* (one per channel). This way we save one memcpy operation and we
* don't need to rely on cudaMemcpy2D. The last preprocessing
* operation will write the separate channels directly to the input
* layer. */
void CTextRecognition::WrapInputLayer(caffe::shared_ptr<caffe::Blob<float>> input_blob, std::vector<cv::Mat>& input_channels) {
	//Blob<float>* input_layer = net_->input_blobs()[0];
	caffe::shared_ptr<caffe::Blob<float>> input_layer = input_blob;
	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}
}

void CTextRecognition::Preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels, int target_width, int target_height) {
	// Convert the input image to the input image format of the network.
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	// processing imgs
	cv::Mat sample_resized;
	cv::Size input_geometry_new = cv::Size(target_width, target_height);
	//cv::resize(sample, sample_resized, input_geometry_new);
	cv::resize(sample, sample_resized, input_geometry_new, (0, 0), (0, 0), INTER_AREA);
	//cv::imwrite("../tmp/stn_input_img.jpg", sample_resized);

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_iv;
	sample_iv = sample_float*(1.0 / 255);

	cv::split(sample_iv, input_channels);
	//Mat outimg;
	//cv::normalize(input_channels[0], outimg, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	//cv::imwrite("../tmp/stn_input_img_0.jpg", outimg);
	//FileStorage fs("../tmp/stn_input_img_0.xml", FileStorage::WRITE);
	//fs << "vocabulary" << input_channels[0];
	//fs.release();
	//CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
}

void CTextRecognition::Predict(const cv::Mat& img, std::vector<float>& output_prob, const string& input_name, const string& output_name) {
	boost::shared_ptr<Blob<float>> input_layer = net_->blob_by_name(input_name);
	int height = img.size().height;
	int width = img.size().width;
	int new_height = input_geometry_.height;
	int new_width = int(1.0*new_height*width / height);

	input_layer->Reshape(1, num_channels_, new_height, new_width);
	net_->Reshape();
	std::vector<cv::Mat> input_channels;
	WrapInputLayer(input_layer, input_channels);
	Preprocess(img, input_channels, new_width, new_height);
	//PrintMat(input_channels[0]);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	boost::shared_ptr<caffe::Blob<float> > output_layer = net_->blob_by_name(output_name);
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->count();
	output_prob.clear();
	output_prob = std::vector<float>(begin, end);
}

// Return the top N predictions.
void CTextRecognition::Recognize(const cv::Mat& img, vector<string>& out_text, int N) {
	const string PADDING = "6862";

    clock_t start1,t_cl;
    start1 = clock();

    std::vector<float> output_prob;
	Predict(img, output_prob, string("data"), string("reshape2"));

    t_cl = clock();
    DEBUG_PRINT(<< "Foward Time: " << 1.0*(t_cl-start1)/CLOCKS_PER_SEC << "s " << endl);

	boost::shared_ptr<caffe::Blob<float> > output_layer = net_->blob_by_name("reshape2");
	int time_steps = output_layer->num();
	int voc_size = output_layer->count() / time_steps;
	//DEBUG_PRINT(<< "time steps is " << time_steps << " , voc size is " << voc_size  << " count " << output_layer->count() << endl);

    const int prob_count = output_prob.size();
    //DEBUG_PRINT(<< "prob_count is: " << prob_count << std::endl);
    std::vector<float> small_add(prob_count, 1e-20);
    std::vector<float> output_add(prob_count);
    caffe_add(prob_count, &output_prob[0], &small_add[0], &output_add[0]);
    std::vector<float> output(prob_count);
    caffe_log(prob_count, &output_add[0], &output[0]);

    std::vector<float> temp;
    std::vector<int> maxid;
    std::vector<int> maxpathid;
    std::vector<string> prevpred;
    std::vector<string> newpred;
    std::vector<float> prevscore;


    for(int k=0; k < voc_size; k++){
       temp.push_back(output[k]);
    }
    maxid = Argmax(temp, N);
    for(int i = 0; i < maxid.size(); i++){
       prevscore.push_back(output[maxid[i]]);
       std::stringstream key;
       key << maxid[i];
       prevpred.push_back(key.str());
    }

    std::vector<int> curlabel;
    std::vector<float> curscore;
    std::vector<float> pathscore;
    std::vector<string> preduck(N);

	//std::string content;
	//for(int i = 0; i < time_steps * voc_size; i = i + voc_size) {
	//   temp.resize(0);
	//   curlabel.resize(0);
	//   curscore.resize(0);
	//   pathscore.resize(0);
	//   for (int j = 0; j < voc_size; j++) {
	//       temp.push_back(output[i+j]);
	//   }
	//   maxid = Argmax(temp, N);
	//   for(int k = 0; k < maxid.size(); k++){
	//       curscore.push_back(temp[maxid[k]]);
	//       curlabel.push_back(maxid[k]);

	//	   pathscore.push_back(temp[maxid[k]]);
	//	   std::stringstream key;
	//	   key << curlabel[0];
	//	   content = content + key.str()+ " ";
	//   }
	//}
	//DEBUG_PRINT(<< content << std::endl);

    for(int i = voc_size; i < time_steps * voc_size; i = i + voc_size) {
       temp.resize(0);
       curlabel.resize(0);
       curscore.resize(0);
       pathscore.resize(0);
       newpred.resize(0);

       for (int j = 0; j < voc_size; j++) {
           temp.push_back(output[i+j]);
       }
       maxid = Argmax(temp, N);
       for(int k = 0; k < maxid.size(); k++) {
           curscore.push_back(temp[maxid[k]]);
           curlabel.push_back(maxid[k]);
       }

       for (int m = 0; m < maxid.size(); m++) {
           for(int n = 0; n < N; n++){
               pathscore.push_back(prevscore[m]+curscore[n]);
           }
       }

       maxpathid = Argmax(pathscore, N);
       for(int q = 0; q < N; q++) {
           std::stringstream key;
           key << curlabel[maxpathid[q]%N];
           newpred.push_back(prevpred[maxpathid[q]/N] + " " + key.str());
       }

       for(int q=0; q < N; q++) {
           prevpred[q] = newpred[q];
           prevscore[q] = pathscore[maxpathid[q]];
       }
    }

	//for(int i=0; i<N; i++){
	//	/*DEBUG_PRINT(<< "key_len:" << newpred[i].length() << " key: " << newpred[i] << endl);*/
	//	DEBUG_PRINT(<< newpred[i] << endl);
	//}

    for(int i=0; i<N; i++) {
       std::istringstream f(newpred[i]);
       string s;
	   string prev = PADDING;
	   
       int x;
       while (getline(f, s, ' ')) {
           if(s!=prev && s!= PADDING) 
		   {
               stringstream(s)>>x;
               preduck[i] = preduck[i] + labels_[x];
			   //DEBUG_PRINT(<< " " << x);
			   //stringstream(prev) >> x;
			   //DEBUG_PRINT(<< " " << x);
           }
           prev = s;
       }
	   //string gbk_str = Utf8ToGbk(preduck[i]);
	   //DEBUG_PRINT(<< "preduck: " << gbk_str << std::endl);
    }

    std::vector<float> finalscore;
    std::vector<string> finalpreds_tmp;
    finalpreds_tmp.push_back(preduck[0]);

    for(int i = 0; i < preduck.size(); i++) {
        float tmp = -100;
        if(i == 0) {
            for(int j = i + 1; j < preduck.size(); j++) {
                if(preduck[i] == preduck[j]) {
                    tmp = log(exp(tmp) + exp(prevscore[i]));
                }
            }
            finalscore.push_back(tmp);
        }
		else {
            bool flag = true;
            for(int k = 0; k < i; k++) {
                if(preduck[i] == preduck[k]) {
                    flag = false;
                    break;
                }
            }
            if(flag == true) {
                finalpreds_tmp.push_back(preduck[i]);
                for(int j = i + 1; j < preduck.size(); j++) {
                    if(preduck[i] == preduck[j]) {
                        tmp = log(exp(tmp) + exp(prevscore[i]));
                    }
                }
                if (i == preduck.size() - 1)  tmp = prevscore[i];
                finalscore.push_back(tmp);
            }
        }
    }

    std::stringstream ss_res;
    std::vector<int> rank = Argmax(finalscore, N);
    for (size_t i=0; i<rank.size(); i++) {
      if (preduck[rank[i]] != "") {
          //ss_res << finalpreds_tmp[rank[i]] << ",";
          out_text.push_back(finalpreds_tmp[rank[i]]);
      }
    }
}

// Return the top N predictions.
void CTextRecognition::RecognizeSTN(const cv::Mat& img, vector<string>& out_text, int N) {
	//const string PADDING = "7134";
	//const string prev = "21018";
	const string PADDING = "6862";

	clock_t start_t, end_t;
	start_t = clock();

	std::vector<float> output_prob;
	cv::Mat loc_img;
	Transform(img, loc_img, "data", "st/theta");
	Predict(img, output_prob, string("stn_data"), string("reshape2"));
	//static int count = 0;
	//count++;
	////if(count==1)
	//{
	//	boost::shared_ptr<caffe::Blob<float> > st_output_layer = net_->blob_by_name("st/st_output");
	//	std::vector<float> v_float;
	//	ConvertBlobToMat(st_output_layer, v_float, loc_img);
	//	string newName = "../tmp/loc_img_" + to_string(count) + ".jpg";
	//	cv::imwrite(newName, loc_img);
	//}
	//{
	//	boost::shared_ptr<caffe::Blob<float> > seg_output_layer = net_->blob_by_name("seg_output");
	//	std::vector<float> v_float;
	//	ConvertBlobToMat(seg_output_layer, v_float, loc_img);
	//	string newName = "../tmp/seg_img_" + to_string(count) + ".jpg";
	//	cv::imwrite(newName, loc_img);
	//}

	end_t = clock();
	DEBUG_PRINT(<< "c++ recognition time: " << 1.0*(end_t - start_t) / CLOCKS_PER_SEC << "s " << endl);

	boost::shared_ptr<caffe::Blob<float> > output_layer = net_->blob_by_name("reshape2");
	int time_steps = output_layer->num();
	int voc_size = output_layer->count() / time_steps;

	const int prob_count = output_prob.size();
	std::vector<float> small_add(prob_count, 1e-20);
	std::vector<float> output_add(prob_count);
	caffe_add(prob_count, &output_prob[0], &small_add[0], &output_add[0]);
	std::vector<float> output(prob_count);
	caffe_log(prob_count, &output_add[0], &output[0]);

	//std::vector<float> output = output_prob;

	std::vector<float> temp;
	std::vector<int> maxid;
	std::vector<int> maxpathid;
	std::vector<string> prevpred;
	std::vector<string> newpred;
	std::vector<float> prevscore;


	for (int k = 0; k < voc_size; k++) {
		temp.push_back(output[k]);
	}
	maxid = Argmax(temp, N);
	for (int i = 0; i < maxid.size(); i++) {
		prevscore.push_back(output[maxid[i]]);
		std::stringstream key;
		key << maxid[i];
		prevpred.push_back(key.str());
	}

	std::vector<int> curlabel;
	std::vector<float> curscore;
	std::vector<float> pathscore;
	std::vector<string> preduck(N);

	//std::string content;
	//for(int i = 0; i < time_steps * voc_size; i = i + voc_size) {
	//   temp.resize(0);
	//   curlabel.resize(0);
	//   curscore.resize(0);
	//   pathscore.resize(0);
	//   for (int j = 0; j < voc_size; j++) {
	//       temp.push_back(output[i+j]);
	//   }
	//   maxid = Argmax(temp, N);
	//   for(int k = 0; k < maxid.size(); k++){
	//       curscore.push_back(temp[maxid[k]]);
	//       curlabel.push_back(maxid[k]);

	//	   pathscore.push_back(temp[maxid[k]]);
	//	   std::stringstream key;
	//	   key << curlabel[0];
	//	   content = content + key.str()+ " ";
	//   }
	//}
	//DEBUG_PRINT(<< content << std::endl);

	for (int i = voc_size; i < time_steps * voc_size; i = i + voc_size) {
		temp.resize(0);
		curlabel.resize(0);
		curscore.resize(0);
		pathscore.resize(0);
		newpred.resize(0);

		for (int j = 0; j < voc_size; j++) {
			temp.push_back(output[i + j]);
		}
		maxid = Argmax(temp, N);
		for (int k = 0; k < maxid.size(); k++) {
			curscore.push_back(temp[maxid[k]]);
			curlabel.push_back(maxid[k]);
		}

		for (int m = 0; m < maxid.size(); m++) {
			for (int n = 0; n < N; n++) {
				pathscore.push_back(prevscore[m] + curscore[n]);
			}
		}

		maxpathid = Argmax(pathscore, N);
		for (int q = 0; q < N; q++) {
			std::stringstream key;
			key << curlabel[maxpathid[q] % N];
			newpred.push_back(prevpred[maxpathid[q] / N] + " " + key.str());
		}

		for (int q = 0; q < N; q++) {
			prevpred[q] = newpred[q];
			prevscore[q] = pathscore[maxpathid[q]];
		}
	}

	//for(int i=0; i<N; i++){
	//	/*DEBUG_PRINT(<< "key_len:" << newpred[i].length() << " key: " << newpred[i] << endl);*/
	//	DEBUG_PRINT(<< newpred[i] << endl);
	//}

	for (int i = 0; i < N; i++) {
		std::istringstream f(newpred[i]);
		string s;
		string prev = PADDING;
		int x;
		while (getline(f, s, ' ')) {
			if (s != prev && s != PADDING)
			{
				stringstream(s) >> x;
				preduck[i] = preduck[i] + labels_[x];
				//DEBUG_PRINT(<< " " << x);
				//stringstream(prev) >> x;
				//DEBUG_PRINT(<< " " << x);
			}
			prev = s;
		}
		//string gbk_str = Utf8ToGbk(preduck[i]);
		//DEBUG_PRINT(<< "preduck: " << gbk_str << std::endl);
	}

	std::vector<float> finalscore;
	std::vector<string> finalpreds_tmp;
	finalpreds_tmp.push_back(preduck[0]);

	for (int i = 0; i < preduck.size(); i++) {
		float tmp = -100;
		if (i == 0) {
			for (int j = i + 1; j < preduck.size(); j++) {
				if (preduck[i] == preduck[j]) {
					tmp = log(exp(tmp) + exp(prevscore[i]));
				}
			}
			finalscore.push_back(tmp);
		}
		else {
			bool flag = true;
			for (int k = 0; k < i; k++) {
				if (preduck[i] == preduck[k]) {
					flag = false;
					break;
				}
			}
			if (flag == true) {
				finalpreds_tmp.push_back(preduck[i]);
				for (int j = i + 1; j < preduck.size(); j++) {
					if (preduck[i] == preduck[j]) {
						tmp = log(exp(tmp) + exp(prevscore[i]));
					}
				}
				if (i == preduck.size() - 1)  tmp = prevscore[i];
				finalscore.push_back(tmp);
			}
		}
	}

	std::stringstream ss_res;
	std::vector<int> rank = Argmax(finalscore, N);
	for (size_t i = 0; i < rank.size(); i++) {
		if (preduck[rank[i]] != "") {
			//ss_res << finalpreds_tmp[rank[i]] << ",";
			out_text.push_back(finalpreds_tmp[rank[i]]);
		}
	}
}

bool CTextRecognition::LoadDict(const string& label_file)
{
    label_file_ = label_file;
    /* Load labels. */
    std::ifstream labels(label_file_.c_str());

    if (!labels) {
		DEBUG_PRINT (<< "Unable to open labels file ." << endl);
        return false;
    }
    //CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line)) {
        labels_.push_back(string(line));
    }

    return true;
}

void CTextRecognition::Init(const string& label_file)
{
	//SetThresh(cf_thresh, ns_thresh);
	LoadDict(label_file);
}

bool CTextRecognition::LoadModel(const string & model_file, const string & weights_file, const string& deviceName)
{
    DEBUG_PRINT( <<" CTextRecognition loadmodel." << endl);

    if (!LoadModel(model_file, weights_file))
	{
      return false;
    }

    //input_geometry_ = cv::Size(320, 32);
	input_geometry_ = cv::Size(320, 64);

    return true;
}

int  CTextRecognitionCache::CreateDetector()
 {
	 int pre_size = detectors_.size();
	 int curr_size = pre_size;
	 bool new_loaded = false;
	 if (curr_size < max_detector_num_)
	 {
		 {
			 string::size_type pos;
			 pos = GetWeightFile().rfind(".caffemodel");
			 if (pos != string::npos)
			 {
				 string model_file = GetModelFile(); string weights_file = GetWeightFile(); string deviceName = GetDeviceName();
				 const vector<string>& deviceIndexes = GetDeviceIndexes();
				 CTextRecognition* entity = new CTextRecognition(model_file, weights_file, deviceIndexes[curr_size]);
				 bool bRet = entity->LoadModel(model_file, weights_file, deviceIndexes[curr_size]);
				 if (!bRet)
				 {
					 DEBUG_PRINT(<< "loadModel failed " << endl);
					 delete entity;
					 return NULL;;
				 }

				 string lable_name = GetLabelFile();
				 entity->Init(lable_name);

				 entity->SetFreeFlag(true);
				 entity->SetInitialised(true);

				 detectors_.push_back(entity);
				 DEBUG_PRINT(<< "CTextRecognitionCache::CreateDetectors " << curr_size << deviceIndexes[curr_size] << endl);
				 new_loaded = true;
			 }
			 detector_num_ = detectors_.size();
			 if (new_loaded)
				 return pre_size;
		 }
	 }

	 return -1;
 }

void CTextRecognitionCache::InitCache(const string& model_file, const string& weights_file, const string& deviceName, const string& label_file, float conf_thresh, float nms_thresh,
	int max_detector_num)
{
	DetectorCache::InitCache(model_file, weights_file, deviceName, max_detector_num);

	label_file_ = label_file;
	//conf_thresh_ = conf_thresh;
	//nms_thresh_ = nms_thresh;
	
}

 static bool GreaterSort(TObject a, TObject b) { return (a.y < b.y); }

 static Mat ConvertMode(Mat& input_img, int channels)
 {
	 Mat sample;
	 if (input_img.channels() == 3 && channels == 1)
		 cv::cvtColor(input_img, sample, cv::COLOR_BGR2GRAY);
	 else if (input_img.channels() == 4 && channels == 1)
		 cv::cvtColor(input_img, sample, cv::COLOR_BGRA2GRAY);
	 else if (input_img.channels() == 4 && channels == 3)
		 cv::cvtColor(input_img, sample, cv::COLOR_BGRA2BGR);
	 else if (input_img.channels() == 1 && channels == 3)
		 cv::cvtColor(input_img, sample, cv::COLOR_GRAY2BGR);
	 else
		 sample = input_img;

	 return sample;
 }

 int CTextRecognitionCache::DoJob(vector<TObject>& boxes, vector<vector<string>>& output_text, int N)
 {
	 output_text.resize(boxes.size());

	 int detector_index = RequestDetector();
	 if (detector_index < 0)
	 {
		 DEBUG_PRINT(<< "logo_getDetector failed. " << endl);
		 return -1;
	 }
	 CTextRecognition* working_detect = (CTextRecognition*)GetDetector(detector_index);
	 if (working_detect)
	 {
		 //std::sort(boxes.begin(), boxes.end(), GreaterSort);
		 for (int j = 0; j < boxes.size(); j++)
		 {
			cv::Mat curr_image = boxes[j].im;

			vector<string> recognized_strings;
			//Mat converted_img = ConvertMode(dst_image, 3);
			working_detect->RecognizeSTN(curr_image, recognized_strings, N);
			output_text[j] = recognized_strings;
		 }
		 FreeDetector(detector_index);
	 }

	 return 0;
 }

 int CTextRecognitionCache::DoJob(cv::Mat image, vector<string>& output_text)
 {
	 int detector_index = RequestDetector();
	 if (detector_index < 0)
	 {
		 DEBUG_PRINT(<< "logo_getDetector failed. " << endl);
		 return -1;
	 }
	 CTextRecognition* working_detect = (CTextRecognition*)GetDetector(detector_index);
	 if (working_detect)
	 {
		working_detect->Recognize(image, output_text, 1);
		 
		FreeDetector(detector_index);
	 }

	 return 0;
 }


 // initialization static variables out of class
 TextRecognitionCacheSingleton::Ptr TextRecognitionCacheSingleton::m_instance_ptr = nullptr;
 std::mutex TextRecognitionCacheSingleton::m_mutex;