#ifndef _DETECTORBASE_H_
#define _DETECTORBASE_H_

#include "common.h"
#include <mutex>         // std::mutex, std::unique_lock
#include "utils.h"

#if defined(_MSC_VER)
#include "EThreadingWindows.h"
/*
*  Global factory for creating synchronization objects
*/
extern FSynchronizeFactory* GSynchronizeFactory;
#endif 

//#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#endif  // USE_OPENCV

using namespace std;

struct TObject {
	int x;
	int y;
	int width;
	int height;
	int class_id;
	float conf;
	cv::Mat im;

	TObject()
	{
		x = 0;
		y = 0;
		width = 0;
		height = 0;
		class_id = -1;
		conf = 0.0;
	}

	TObject& operator=(const TObject& obj)
	{
		if (this != &obj)
		{
			x = obj.x;
			y = obj.y;
			width = obj.width;
			height = obj.height;
			class_id = obj.class_id;
			conf = obj.conf;
			im = obj.im;
		}
		return *this;
	}

};

template <typename DTYPE>
inline void createTObject(const TemRect<DTYPE>& in_rect, TObject& out_body, float scale = 1.0)
{
	out_body.x = in_rect.x*scale;
	out_body.y = in_rect.y*scale;
	out_body.width = in_rect.width*scale;
	out_body.height = in_rect.height*scale;
	out_body.conf = in_rect.score;
}

inline bool test_collision(const TObject& rect1, const TObject& rect2)
{
	int x1 = rect1.x;
	int y1 = rect1.y;
	int x2 = rect1.x + rect1.width;
	int y2 = rect1.y + rect1.height;

	int x3 = rect2.x;
	int y3 = rect2.y;
	int x4 = rect2.x + rect2.width;
	int y4 = rect2.y + rect2.height;

	return (((x1 >= x3 && x1 < x4) || (x3 >= x1 && x3 <= x2)) &&
		((y1 >= y3 && y1 < y4) || (y3 >= y1 && y3 <= y2))) ? true : false;

}


class DetectorBase
{
public:
    DetectorBase(const string& model_file, const string& weights_file, const string& deviceName);
    virtual ~DetectorBase();

    virtual bool LoadModel(const string & model_file, const string & weights_file, const string& deviceName) = 0;

    inline bool Initialised(){ return initialized_;}
    inline void SetInitialised(bool flag){ initialized_=flag;}
    inline bool IsFree()const{ return free_;}
    inline void SetFreeFlag(bool flag){free_ = flag;}

private:
    bool             initialized_;
    volatile bool    free_;
    string           model_file_;
    string           weights_file_;
    string           deviceName_;
};

class DetectorCache
{
public:
    DetectorCache();
    virtual ~DetectorCache();
public:
    virtual int             RequestDetector();
    virtual void            FreeDetector(int index);
	virtual void            ReleaseDetectors();
	virtual int             CreateDetector() = 0;
	virtual void            InitCache(const string& model_file, const string& weights_file, const string& deviceName, int max_detector_num = 1);

public:
    inline vector<DetectorBase*>& GetDetectors()
    {
        return detectors_;
    }
    inline DetectorBase* GetDetector(int index) const
    {
        return detectors_[index];
    }
    inline int GetDetectorNum() const
    {
        return detector_num_;
    }
    inline std::string GetModelFile() const
    {
        return model_file_;
    }
    inline std::string GetWeightFile() const
    {
        return weights_file_;
    }
    inline std::string GetDeviceName() const
    {
        return deviceName_;
    }
    inline const vector<string>& GetDeviceIndexes() const
    {
        return deviceIndexes_;
    }
    inline int GetMaxDetectorNum() const
    {
        return max_detector_num_;
    }

protected:
    vector<DetectorBase*> detectors_;
    int                   detector_num_;
    int                   max_detector_num_;
    string                model_file_;
    string                weights_file_;
    string                deviceName_;
    vector<string>        deviceIndexes_;

    #if defined(_MSC_VER)
	FCriticalSection*     criticalSection_;
	FEvent*               synchEvent_;
    #else
    pthread_mutex_t       logo_detector_mutex;
    pthread_cond_t        logo_detector_free_cond;
    #endif
};

#endif
