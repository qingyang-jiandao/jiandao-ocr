#ifndef TEXT_RECOGNIZATION_H
#define TEXT_RECOGNIZATION_H

#include "detectorBase.h"
//#include "caffe_public_api.h"
#include <caffe/caffe.hpp>

class CTextRecognition : public DetectorBase
{
public:
    CTextRecognition(const string& model_file, const string& weights_file, const string& deviceName, 
		const string& label_file=string());
    ~CTextRecognition();

public:
    virtual bool LoadModel(const string & model_file, const string & weights_file, const string& deviceName);
	void Predict(const cv::Mat& img, vector<float>& output_prob, const string& input_name=string("data"), const string& output_name = string("reshape2"));
	void WrapInputLayer(caffe::shared_ptr<caffe::Blob<float>> input_blob, vector<cv::Mat>& input_channels);
	void Preprocess(const cv::Mat& img, vector <cv::Mat>& input_channels, int target_width, int target_height);
	bool LoadModel(const string& model_file, const string& trained_file);
	void Transform(const cv::Mat& img, cv::Mat& loc_img, const string& input_name, const string& output_name);
	bool LoadDict(const string& label_file);
	void Init(const string& label_file);
	void Recognize(const cv::Mat& img, vector<string>& out_text, int N = 1);
	void RecognizeSTN(const cv::Mat& img, vector<string>& out_text, int N = 1);

private:
	caffe::shared_ptr<caffe::Net<float>> net_;
    cv::Size input_geometry_;
    int num_channels_;
    string label_file_;
    vector<string> labels_;
};

class CTextRecognitionCache : public DetectorCache
{
protected:
	virtual int CreateDetector();
	inline void SetLabelFile(const string label_file)
	{
		label_file_ = label_file;
	}
	inline string GetLabelFile() const
	{
		return label_file_;
	}
	//inline void SetThresh(float cf_thresh, float ns_thresh)
	//{
	//	conf_thresh_ = cf_thresh;
	//	nms_thresh_ = ns_thresh;
	//}
	//inline void GetThresh(float& out_cf_thresh, float& out_ns_thresh)
	//{
	//	out_cf_thresh = conf_thresh_;
	//	out_ns_thresh = nms_thresh_;
	//}
	//inline void SetLabelFile(const string label_file)
	//{
	//	label_file_ = label_file;
	//}
	//inline string GetLabelFile() const
	//{
	//	return label_file_;
	//}
public:
	void InitCache(const string& model_file, const string& weights_file, const string& deviceName, const string& label_file, float conf_thresh = 0.8, float nms_thresh = 0.3, int max_detector_num = 1);
    int DoJob(vector<TObject>& boxes, vector<vector<string>>& output_text, int N=1);
	int DoJob(cv::Mat image, vector<string>& output_text);

protected:
	string label_file_;
};

class TextRecognitionCacheSingleton {
public:
	typedef CTextRecognitionCache* Ptr;
	~TextRecognitionCacheSingleton() {
		std::cout << "DetectorCache destructor called!" << std::endl;
	}
	TextRecognitionCacheSingleton(TextRecognitionCacheSingleton&) = delete;
	TextRecognitionCacheSingleton& operator=(const TextRecognitionCacheSingleton&) = delete;
	static Ptr get_instance(const string& cfgfile, const string& weightfile, const string& deviceName, const string& label_file, float conf_thresh, float nms_thresh, int max_detector_num) {

		// "double checked lock"
		if (m_instance_ptr == nullptr) {
			std::lock_guard<std::mutex> lk(m_mutex);
			if (m_instance_ptr == nullptr) {
				m_instance_ptr = new CTextRecognitionCache;
				m_instance_ptr->InitCache(cfgfile, weightfile, deviceName, label_file, conf_thresh, nms_thresh, max_detector_num);
			}
		}
		return m_instance_ptr;
	}

	static Ptr& get_instance() {
		return m_instance_ptr;
	}


private:
	TextRecognitionCacheSingleton() {
		std::cout << "DetectorCache constructor called!" << std::endl;
	}
	static Ptr m_instance_ptr;
	static std::mutex m_mutex;
};

#endif // TEXT_RECOGNIZATION_H
