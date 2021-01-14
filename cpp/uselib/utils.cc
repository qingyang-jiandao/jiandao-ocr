#include "utils.h"

#ifdef _MSC_VER
//#include <cstring>        // for strcpy(), strcat()
#else
#include <fcntl.h>
#include <unistd.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/dir.h>
#endif

namespace RON{
    void nms(std::vector<abox> &input_boxes, float nms_thresh){
        std::vector<float>vArea(input_boxes.size());
        for (int i = 0; i < input_boxes.size(); ++i)
        {
            vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
        }
        for (int i = 0; i < input_boxes.size(); ++i)
        {
            for (int j = i + 1; j < input_boxes.size();)
            {
                float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
                float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
                float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
                float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
                float w = std::max(float(0), xx2 - xx1 + 1);
                float   h = std::max(float(0), yy2 - yy1 + 1);
                float   inter = w * h;
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= nms_thresh)
                {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else
                {
                    j++;
                }
            }
        }
    }
}

#ifdef _MSC_VER
#include <iostream>
#include <Windows.h>
#include <io.h>

int isFolderExist(const string& path)
{
	if (_access(path.c_str(), 0) != -1)
	{
		return 1;
	}
	return 0;
}


bool is_dir(const std::string& file_name)
{
	struct stat file_info;
	if (stat(file_name.c_str(), &file_info)==0)
	{
		if (file_info.st_mode & S_IFDIR)
		{
			return true;
		}
	}

	return false;
}

void listFiles(string path, vector<string>& out_files)
{
	intptr_t handle;
	_finddata_t findData;
	string pathName, exdName;

	handle = _findfirst(pathName.assign(path).append("\\*").c_str(), &findData);
	if (handle == -1)
		return;
	do
	{
		if (findData.attrib & _A_SUBDIR)
		{
			string fname = string(findData.name);
			if (fname != ".." && fname != ".") {
				//cout << findData.name << "\n";
				listFiles(path + "\\" + fname, out_files);
			}
		}
		else {
			//cout << findData.name << "\t" << findData.size << " bytes.\n";
			out_files.push_back(path + "\\" + findData.name);
		}
	} while (_findnext(handle, &findData) == 0);

	_findclose(handle);
}

//couvert gbk to UTF-8
std::string GbkToUtf8(const std::string& strGbk)
{
	int len = MultiByteToWideChar(CP_ACP, 0, strGbk.c_str(), -1, NULL, 0);
	wchar_t *strUnicode = new wchar_t[len];
	wmemset(strUnicode, 0, len);
	MultiByteToWideChar(CP_ACP, 0, strGbk.c_str(), -1, strUnicode, len);

	//couvert unicode to UTF-8
	len = WideCharToMultiByte(CP_UTF8, 0, strUnicode, -1, NULL, 0, NULL, NULL);
	char * strUtf8 = new char[len];
	WideCharToMultiByte(CP_UTF8, 0, strUnicode, -1, strUtf8, len, NULL, NULL);

	std::string strTemp(strUtf8);
	delete[]strUnicode;
	delete[]strUtf8;
	strUnicode = NULL;
	strUtf8 = NULL;

	return strTemp;
}

//couvert UTF-8 to gbk
std::string Utf8ToGbk(const std::string& strUtf8)
{
	int len = MultiByteToWideChar(CP_UTF8, 0, strUtf8.c_str(), -1, NULL, 0);
	wchar_t * strUnicode = new wchar_t[len];
	wmemset(strUnicode, 0, len);
	MultiByteToWideChar(CP_UTF8, 0, strUtf8.c_str(), -1, strUnicode, len);

	len = WideCharToMultiByte(CP_ACP, 0, strUnicode, -1, NULL, 0, NULL, NULL);
	char *strGbk = new char[len];
	memset(strGbk, 0, len);
	WideCharToMultiByte(CP_ACP, 0, strUnicode, -1, strGbk, len, NULL, NULL);

	std::string strTemp(strGbk);
	delete[]strUnicode;
	delete[]strGbk;
	strUnicode = NULL;
	strGbk = NULL;

	return strTemp;
}

//couvert gbk to unicode
std::wstring GbkToUnicode(const std::string& strGbk)
{
	int len = MultiByteToWideChar(CP_ACP, 0, strGbk.c_str(), -1, NULL, 0);
	wchar_t *strUnicode = new wchar_t[len];
	wmemset(strUnicode, 0, len);
	MultiByteToWideChar(CP_ACP, 0, strGbk.c_str(), -1, strUnicode, len);

	std::wstring strTemp(strUnicode);
	delete[]strUnicode;
	strUnicode = NULL;
	return strTemp;
}

//couvert unicode to gbk
std::string UnicodeToGbk(const std::wstring& strUnicode)
{
	int len = WideCharToMultiByte(CP_ACP, 0, strUnicode.c_str(), -1, NULL, 0, NULL, NULL);
	char *strGbk = new char[len];
	memset(strGbk, 0, len);
	WideCharToMultiByte(CP_ACP, 0, strUnicode.c_str(), -1, strGbk, len, NULL, NULL);

	std::string strTemp(strGbk);
	delete[]strGbk;
	strGbk = NULL;
	return strTemp;
}

#else

void server_backtrace(int sig)
{
    //打开文件
    time_t tSetTime;
    time(&tSetTime);
    struct tm* ptm = localtime(&tSetTime);
    char fname[256] = {0};
    sprintf(fname, "core.%d-%d-%d_%d_%d_%d",
            ptm->tm_year+1900, ptm->tm_mon+1, ptm->tm_mday,
            ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
    FILE* f = fopen(fname, "a");
    if (f == NULL){
        return;
    }
    int fd = fileno(f);

    //锁定文件
    struct flock fl;
    fl.l_type = F_WRLCK;
    fl.l_start = 0;
    fl.l_whence = SEEK_SET;
    fl.l_len = 0;
    fl.l_pid = getpid();
    fcntl(fd, F_SETLKW, &fl);

    //输出程序的绝对路径
    char buffer[4096];
    memset(buffer, 0, sizeof(buffer));
    int count = readlink("/proc/self/exe", buffer, sizeof(buffer));
    if(count > 0){
        buffer[count] = '\n';
        buffer[count + 1] = 0;
        fwrite(buffer, 1, count+1, f);
    }

    //输出信息的时间
    memset(buffer, 0, sizeof(buffer));
    sprintf(buffer, "Dump Time: %d-%d-%d %d:%d:%d\n",
            ptm->tm_year+1900, ptm->tm_mon+1, ptm->tm_mday,
            ptm->tm_hour, ptm->tm_min, ptm->tm_sec);
    fwrite(buffer, 1, strlen(buffer), f);

    //线程和信号
    sprintf(buffer, "Curr thread: %u, Catch signal:%d\n",
            (int)pthread_self(), sig);
    fwrite(buffer, 1, strlen(buffer), f);

    //堆栈
    void* DumpArray[256];
    int    nSize =    backtrace(DumpArray, 256);
    sprintf(buffer, "backtrace rank = %d\n", nSize);
    fwrite(buffer, 1, strlen(buffer), f);
    if (nSize > 0){
        char** symbols = backtrace_symbols(DumpArray, nSize);
        if (symbols != NULL){
            for (int i=0; i<nSize; i++){
                fwrite(symbols[i], 1, strlen(symbols[i]), f);
                fwrite("\n", 1, 1, f);
            }
            free(symbols);
        }
    }

    //文件解锁后关闭
    fl.l_type = F_UNLCK;
    fcntl(fd, F_SETLK, &fl);
    fclose(f);
}

void signal_crash_handler(int sig)
{
    server_backtrace(sig);
    exit(-1);
}

void signal_exit_handler(int sig)
{
    exit(0);
}

void server_on_exit(void)
{
    //do something when process exits
}

void trace_stack(void)
{
    atexit(server_on_exit);
    signal(SIGTERM, signal_exit_handler);
    signal(SIGINT, signal_exit_handler);

    // ignore SIGPIPE
    signal(SIGPIPE, SIG_IGN);

    signal(SIGBUS, signal_crash_handler);     // 总线错误
    signal(SIGSEGV, signal_crash_handler);    // SIGSEGV，非法内存访问
    signal(SIGFPE, signal_crash_handler);       // SIGFPE，数学相关的异常，如被0除，浮点溢出，等等
    signal(SIGABRT, signal_crash_handler);     // SIGABRT，由调用abort函数产生，进程非正常退出
}

//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
bool is_dir(const std::string& file_name)
{
    struct stat file_info;
    stat(file_name.c_str(), &file_info);

    if(S_ISDIR(file_info.st_mode))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int isFolderExist(const std::string& path)
{
    DIR* dp;
    if(dp=opendir(path.c_str()))
    {
        closedir(dp);
        return 1;
    }
    return 0;
}

void listFiles(const string path, vector<string>& out_files)
{
	out_files.clear();

	bool dir_flag = is_dir(path);
	if (dir_flag)
	{
		string input_dir = path;
		if (path[path.length()] != '/')
			input_dir = input_dir + "/";

		DIR *dir = NULL;
		struct dirent* ptr;
		if ((dir = opendir(input_dir.c_str())) == NULL)
		{
			DEBUG_PRINT(<< "cate_dir is wrong" << endl);
			return;
		}

		int img_number = 0;
		while ((ptr = readdir(dir)) != NULL)
		{
			if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
			{
				continue;
			}
			else if (ptr->d_type == 10) //link file
			{
				continue;
			}
			else if (ptr->d_type == 4) //dir
			{
				continue;
			}
			else if (ptr->d_type == 8) //file
			{
				img_number++;
				out_files.push_back(input_dir + ptr->d_name);
			}
		}
		if (dir)closedir(dir);
	}
	else
	{
		out_files.push_back(path);
	}

	//    if(out_files.size()>0)
	//        sort(out_files.begin(), out_files.end());
}

#endif

inline string& trim(string &s, const string& mask)
{
    if (s.empty())
    {
        return s;
    }

    s.erase(0,s.find_first_not_of(mask));
    s.erase(s.find_last_not_of(mask) + 1);
    return s;
}

void string_replace(const string& s, const string& oldsub,
    const string& newsub, bool replace_all, string* res) {
    if (oldsub.empty()) {
        res->append(s);  // if empty, append the given string.
        return;
    }

    string::size_type start_pos = 0;
    string::size_type pos;
    do {
        pos = s.find(oldsub, start_pos);
        if (pos == string::npos) {
            break;
        }
        res->append(s, start_pos, pos - start_pos);
        res->append(newsub);
        start_pos = pos + oldsub.size();  // start searching again after the "old"
    } while (replace_all);
    res->append(s, start_pos, s.length() - start_pos);
}

std::vector<std::string> splitString(std::string srcStr, std::string delimStr, bool repeatedCharIgnored)
{
    std::vector<std::string> resultStringVector;
    std::replace_if(srcStr.begin(), srcStr.end(), [&](const char& c) {if (delimStr.find(c) != std::string::npos) { return true; } else { return false; }}/*pred*/, delimStr.at(0));//将出现的所有分隔符都替换成为一个相同的字符（分隔符字符串的第一个）
    size_t pos = srcStr.find(delimStr.at(0));
    std::string addedString = "";
    while (pos != std::string::npos) {
        addedString = srcStr.substr(0, pos);
        if (!addedString.empty() || !repeatedCharIgnored) {
            resultStringVector.push_back(addedString);
        }
        srcStr.erase(srcStr.begin(), srcStr.begin() + pos + 1);
        pos = srcStr.find(delimStr.at(0));
    }
    addedString = srcStr;
    if (!addedString.empty() || !repeatedCharIgnored) {
        resultStringVector.push_back(addedString);
    }
    return resultStringVector;
}

int read_classname_list(const string& filename, vector<string>& classnames, int index)
{
    ifstream fin(filename);
    string classinfo;
    while (getline(fin, classinfo))
    {
        classinfo=trim(classinfo, " ");
        vector<string> subStrings = splitString(classinfo, " ", true);
        string classname = subStrings[index];
        //DEBUG_PRINT(<< classname  << endl);
        classnames.push_back(classname);
    }
    return 0;
}

int read_logoname2text_list(const string& filename, map<string, vector<string>>& logolabel2textlist_map)
{
    ifstream fin(filename);
    string line;
    while (getline(fin, line))
    {
        line=trim(line, " ");
        size_t pos = line.find(" ");
        string logoname = "";
        if (pos != string::npos) {
            logoname = line.substr(0, pos);
            //DEBUG_PRINT(<< logoname  << endl);
            string text_info = line.substr(pos+1, line.length());
            pos = text_info.find("[");
            if (pos != string::npos) {
                string tmp = text_info.substr(pos+1, text_info.length());

                pos = tmp.find("]");
                if (pos != string::npos) {
                    text_info = tmp.substr(0, pos);

                    vector<string> subStrings = splitString(text_info, " ", true);
                    logolabel2textlist_map[logoname] = subStrings;
                }
            }
        }
    }

    return 0;
}

int read_image_list(const string& filename, vector<string>& images)
{
    ifstream fin(filename);
    string imagename;
    while (getline(fin, imagename))
    {
        imagename=trim(imagename, " ");
        //bug
//        string temp;
//        string_replace(imagename, "\n", "", true, &temp);
//        imagename = temp; temp.clear();
//        string_replace(imagename, "\t", "", true, &temp);
//        imagename = temp; temp.clear();
//        string_replace(imagename, "\r", "", true, &temp);
//        imagename = temp; temp.clear();

        if(imagename.length()>0)
        {
            vector<string> subStrings = splitString(imagename, " ", true);
            string dstname = subStrings[0];
            if(dstname.length()>0)
            {
                //DEBUG_PRINT(<< dstname  << endl);
                images.push_back(dstname);
            }
        }
    }
    return 0;
}

int read_image_extendedname_list(const string& filename, vector<string>& extendednames)
{
    ifstream fin(filename);
    string exname;
    while (getline(fin, exname))
    {
        exname=trim(exname, " ");
        string temp;
        string_replace(exname, "\n", "", true, &temp);
        exname = temp; temp.clear();
        string_replace(exname, "\t", "", true, &temp);
        exname = temp; temp.clear();
        string_replace(exname, "\r", "", true, &temp);
        exname = temp; temp.clear();
        //DEBUG_PRINT(<< exname << exname.size() << endl);
        extendednames.push_back(exname);
    }
    return 0;
}

bool is_valid_image(const string& filename, const vector<string>& extendednames)
{
    string::size_type nSize = filename.size();
    string::size_type pos;
    pos = filename.rfind(".");
    if(pos!=string::npos)
    {
        string extname = filename.substr(pos, nSize);
        transform(extname.begin(), extname.end(), extname.begin(), ::tolower);
        vector<string>::const_iterator res = find(extendednames.begin(), extendednames.end(), extname);
        if(res!= extendednames.end())
        {
            //DEBUG_PRINT(<< extname << endl);
            return true;
        }
    }
    return false;
}
