#ifndef UTILS_H
#define UTILS_H

#include <string.h>
#include <vector>
#include <map>
#include <fstream>
#include <algorithm>
#include "common.h"

using namespace std;

//#include <codecvt>

//std::wstring s2ws(const std::string& str)
//{
//    using convert_typeX = std::codecvt_utf8<wchar_t>;
//    std::wstring_convert<convert_typeX, wchar_t> converterX;

//    return converterX.from_bytes(str);
//}

//std::string ws2s(const std::wstring& wstr)
//{
//    using convert_typeX = std::codecvt_utf8<wchar_t>;
//    std::wstring_convert<convert_typeX, wchar_t> converterX;

//    return converterX.to_bytes(wstr);
//}

//#include <cstdlib>
//#include <string>

//inline std::wstring s2ws(const std::string& str) {
//  if (str.empty()) {
//    return L"";
//  }
//  unsigned len = str.size() + 1;
//  setlocale(LC_CTYPE, "en_US.UTF-8");
//  wchar_t *p = new wchar_t[len];
//  mbstowcs(p, str.c_str(), len);
//  std::wstring w_str(p);
//  delete[] p;
//  return w_str;
//}

//inline std::string ws2s(const std::wstring& w_str) {
//    if (w_str.empty()) {
//      return "";
//    }
//    unsigned len = w_str.size() * 4 + 1;
//    setlocale(LC_CTYPE, "en_US.UTF-8");
//    char *p = new char[len];
//    wcstombs(p, w_str.c_str(), len);
//    std::string str(p);
//    delete[] p;
//    return str;
//}


//inline std::wstring s2ws(const std::string& str) {
//  if (str.empty()) {
//    return L"";
//  }
//  unsigned len = str.size() + 1;
//  setlocale(LC_CTYPE, "");
//  wchar_t *p = new wchar_t[len];
//  mbstowcs(p, str.c_str(), len);
//  std::wstring w_str(p);
//  delete[] p;
//  return w_str;
//}
//
//inline std::string ws2s(const std::wstring& w_str) {
//    if (w_str.empty()) {
//      return "";
//    }
//    unsigned len = w_str.size() * 4 + 1;
//    setlocale(LC_CTYPE, "");
//    char *p = new char[len];
//    wcstombs(p, w_str.c_str(), len);
//    std::string str(p);
//    delete[] p;
//    return str;
//}
//

inline void splitUtf8Word(const string& word, vector<string>& characters)
{
    int num = word.size();
    int i = 0;
    while(i < num)
    {
        int size = 1;
        if(word[i] & 0x80)
        {
			//DEBUG_PRINT(<< "xxxxxxxxxxxx Utf8" << endl);
            char temp = word[i];
            temp <<= 1;
            do{
                temp <<= 1;
                ++size;
            }while(temp & 0x80);
        }
        string subWord;
        subWord = word.substr(i, size);
        characters.push_back(subWord);
        i += size;
    }
}

#ifdef _MSC_VER
//couvert gbk to UTF-8
std::string GbkToUtf8(const std::string& strGbk);
//couvert UTF-8 to gbk
std::string Utf8ToGbk(const std::string& strUtf8);
//couvert gbk to unicode
std::wstring GbkToUnicode(const std::string& strGbk);
//couvert unicode to gbk
std::string UnicodeToGbk(const std::wstring& strUnicode);
#endif

#ifndef _MSC_VER
void server_backtrace(int sig);
void signal_crash_handler(int sig);
void signal_exit_handler(int sig);
void server_on_exit(void);
void trace_stack(void);
#endif 


//namespace RPN{
//    struct abox
//    {
//        float x1;
//        float y1;
//        float x2;
//        float y2;
//        float score;
//        bool operator <(const abox&tmp) const{
//            return score < tmp.score;
//        }
//    };
//    cv::Mat bbox_tranform_inv(cv::Mat local_anchors, cv::Mat boxs_delta);
//    void nms(std::vector<abox> &input_boxes, float nms_thresh);


//}

//namespace RPNTEXT_LAB{
//    struct abox
//    {
//        float x1;
//        float y1;
//        float x2;
//        float y2;
//        float score;
//        bool operator <(const abox&tmp) const{
//            return score < tmp.score;
//        }
//    };
//
//    static int box_distance = 50 + 16;
//    void nms(std::vector<abox> &input_boxes, float nms_thresh);
//    void set_box_distance(int value);
//    int  get_box_distance();
//}

namespace RON{
    struct abox
    {
        float x1;
        float y1;
        float x2;
        float y2;
        float score;
        int cls;
        bool operator <(const abox&tmp) const{
            return score < tmp.score;
        }
    };
    void nms(std::vector<abox> &input_boxes, float nms_thresh);
}

//--------------------------------------------------------------------------------
//--------------------------------------------------------------------------------
int isFolderExist(const std::string& path);
bool is_dir(const std::string& file_name);
void listFiles(const string path, vector<string>& out_files);
int read_image_list(const string& filename, vector<string>& images);
int read_classname_list(const string& filename, vector<string>& classnames, int index=0);
int read_logoname2text_list(const string& filename, map<string, vector<string>>& logolabel2textlist_map);
int read_image_extendedname_list(const string& filename, vector<string>& extendednames);
bool is_valid_image(const string& filename, const vector<string>& extendednames);
std::vector<std::string> splitString(std::string srcStr, std::string delimStr, bool repeatedCharIgnored);

#endif  // UTILS_H
