#include "detectorBase.h"
#include "text_recognition.h"

DetectorBase::DetectorBase(const string& model_file, const string& weights_file, const string& deviceName="")
    :model_file_(model_file)
    ,weights_file_(weights_file)
    ,deviceName_(deviceName)
{
    initialized_   = false;
    free_          = false;
}

DetectorBase::~DetectorBase()
{
    initialized_   = false;
    free_          = false;
}

DetectorCache::DetectorCache()
{
    detector_num_ = 0;
    max_detector_num_ = 1;
	#if defined(_MSC_VER)
	criticalSection_ = GSynchronizeFactory->CreateCriticalSection();
	synchEvent_ = GSynchronizeFactory->CreateSynchEvent();
	#else
	pthread_mutex_init(&logo_detector_mutex, NULL);
	pthread_cond_init(&logo_detector_free_cond, NULL);
	#endif
}

DetectorCache::~DetectorCache()
{
    ReleaseDetectors();
}

void DetectorCache::ReleaseDetectors()
{
    if(detectors_.size()>0)
    {
        for(int i=0; i<detectors_.size(); i++)
        {
            delete detectors_[i];
            detectors_[i] = NULL;
        }
    }
    detectors_.clear();
    detector_num_ = 0;
    max_detector_num_ = 1;
}

int DetectorCache::RequestDetector()
{
#if defined(_MSC_VER)

	vector<DetectorBase*>& logo_detectors = GetDetectors();
	int max_num = GetMaxDetectorNum();

	int free_index = -1;

	while (max_num > 0)
	{
		criticalSection_->Lock();

		for (int k = 0; k < logo_detectors.size(); k++)
		{
			if (logo_detectors[k]->IsFree())
			{
				free_index = k;
				break;
			}
		}

		if (free_index >= 0 && free_index < logo_detectors.size())
		{
			logo_detectors[free_index]->SetFreeFlag(false);
			criticalSection_->Unlock();
			break;
		}

		free_index = CreateDetector();

		if (free_index >= 0 && free_index < GetDetectors().size())
		{
			logo_detectors[free_index]->SetFreeFlag(false);
			criticalSection_->Unlock();
			break;
		}

		criticalSection_->Unlock();

		{
			synchEvent_->Wait();
		}
	}//while

	criticalSection_->Unlock();
	return free_index;
#else
		vector<DetectorBase*>& logo_detectors = GetDetectors();
		int max_num = GetMaxDetectorNum();

		int free_index = -1;

		pthread_mutex_lock (&logo_detector_mutex);

		while(max_num >0)
		{
			for(int k=0; k < logo_detectors.size(); k++)
			{
				if (logo_detectors[k]->IsFree())
				{
					free_index = k;
					break;
				}
			}

			if (free_index >= 0 && free_index < logo_detectors.size())
			{
				logo_detectors[free_index]->SetFreeFlag(false);
				break;
			}

			free_index = CreateDetector();

			if (free_index >= 0 && free_index < GetDetectors().size())
			{
				logo_detectors[free_index]->SetFreeFlag(false);
				break;
			}

			{
				pthread_cond_wait(&logo_detector_free_cond, &logo_detector_mutex);
			}

		}//while

		pthread_mutex_unlock(&logo_detector_mutex);
		return free_index;
#endif
}

void DetectorCache::FreeDetector(int index)
{
    if(index < 0) return;

    vector<DetectorBase*>&  logo_detectors = GetDetectors();

	#if defined(_MSC_VER)
		criticalSection_->Lock();

		logo_detectors[index]->SetFreeFlag(true);
		synchEvent_->Trigger();

		criticalSection_->Unlock();

	#else
		pthread_mutex_lock (&logo_detector_mutex);
		logo_detectors[index]->SetFreeFlag(true);
		pthread_mutex_unlock(&logo_detector_mutex);
	#endif
}

void DetectorCache::InitCache(const string& model_file, const string& weights_file, const string& deviceName, 
	int max_detector_num)
{
    model_file_ = model_file;
    weights_file_ = weights_file;
    deviceName_ = deviceName;
    max_detector_num_ = max_detector_num;
    if(max_detector_num_==0)
        max_detector_num_ = 1;
    
	deviceIndexes_.resize(max_detector_num_);
	int gpu_id = -1;
	string delimStr = "gpu:";
	string::size_type pos = deviceName.find(delimStr, 0);
	if (pos != string::npos) {
		string str_gpu_id = deviceName.substr(pos + delimStr.length(), deviceName.length());
		vector<std::string> gpu_ids = splitString(str_gpu_id, ",", true);
		if (gpu_ids.size()<1 || atoi(gpu_ids[0].c_str())<0)
		{
			for(int i=0; i<max_detector_num_; i++)
			{
				deviceIndexes_.push_back("gpu:-1");
			}
		}
		int models_pre_gpu = max_detector_num_/gpu_ids.size();
		int remainder = max_detector_num_%gpu_ids.size();
		for(int i=0; i<gpu_ids.size(); i++)
		{
			for(int j=0; j<models_pre_gpu; j++)
			{
				deviceIndexes_[i*models_pre_gpu+j] = delimStr+gpu_ids[i];
				// cout << "-- " << deviceIndexes_[i*models_pre_gpu+j] << endl;
			}
		}
		for(int j=0; j<remainder; j++)
		{
			deviceIndexes_[models_pre_gpu*gpu_ids.size()+j] = delimStr+gpu_ids[0];
		}
	}
}