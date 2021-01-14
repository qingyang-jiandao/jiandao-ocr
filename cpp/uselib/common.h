#ifndef COMMON_H
#define COMMON_H

#include <string>

//#ifdef ENABLE_DEBUG
//    #define DEBUG_PRINT(format,args...) \
//        printf(format, ##args)
//#else
//    #define DEBUG_PRINT(format,args...)
//#endif

#ifdef ENABLE_DEBUG
    #define DEBUG_PRINT(args) \
        cout args
#else
    #define DEBUG_PRINT(args)
#endif

#define appIsDebuggerPresent	IsDebuggerPresent

inline void appFailAssertFuncDebug(const char* Expr, const char* File, int Line, ...)
{
	printf("%s(%i): Assertion failed: %s\n", File, Line, Expr);
}

#define appFailAssert(expr,file,line,...)				{ if (appIsDebuggerPresent()) appFailAssertFuncDebug(expr, file, line, ##__VA_ARGS__); DebugBreak(); }

#define verify(expr)				{ if(!(expr)) appFailAssert( #expr, __FILE__, __LINE__ ); }

#ifdef __cplusplus
extern "C"
{
#endif

 //

#ifdef __cplusplus
}
#endif

template <typename DTYPE>
struct TemRect{
    DTYPE x;
    DTYPE y;
    DTYPE width;
    DTYPE height;
    DTYPE score;

    TemRect()
    {
        x =  0 ;
        y =  0 ;
        width =  0;
        height =  0 ;
		score = 0;
    }

    explicit TemRect(DTYPE i_x, DTYPE i_y, DTYPE i_width, DTYPE i_height, DTYPE i_pixelNum)
    {
        x =  i_x ;
        y =  i_y;
        width =  i_width;
        height =  i_height ;
		score = i_pixelNum;
    }

    TemRect& operator=(const TemRect& obj)
    {
        if (this != &obj)
        {
            x =  obj.x;
            y =  obj.y;
            width =  obj.width;
            height =  obj.height;
			score = obj.score;
        }
        return *this;
    }

};

typedef TemRect<int> TRect;
typedef TemRect<float> TRectFloat;

template <typename DTYPE>
inline void patch_rect(TemRect<DTYPE>& in_out_rect, int std_width, int std_height, int patch_lefttop, int patch_rightbottom)
{
	DTYPE old_x = in_out_rect.x;
	DTYPE old_y = in_out_rect.y;
	DTYPE patch_left = 0;
	DTYPE patch_top = 0;

    in_out_rect.x = in_out_rect.x - patch_lefttop;
    if (in_out_rect.x < 0 )
        in_out_rect.x = 0;
    patch_left = old_x - in_out_rect.x;

    in_out_rect.y = in_out_rect.y - patch_lefttop;
    if (in_out_rect.y < 0)
        in_out_rect.y = 0;
    patch_top = old_y - in_out_rect.y;

	DTYPE tmp_w =  old_x + in_out_rect.width + patch_rightbottom;
    if(tmp_w>std_width)
        in_out_rect.width=std_width-in_out_rect.x-1;
    else in_out_rect.width = in_out_rect.width + patch_rightbottom+patch_left;

	DTYPE tmp_h =  old_y + in_out_rect.height + patch_rightbottom;
    if(tmp_h>std_height)
        in_out_rect.height=std_height-in_out_rect.y-1;
    else in_out_rect.height=in_out_rect.height + patch_rightbottom+patch_top;
}

inline void pad_rect(TRect& in_out_rect, int max_width, int max_height, int& pad_left, int& pad_right, int& pad_top, int& pad_bottom)
{
	int old_x = in_out_rect.x;
	int old_y = in_out_rect.y;
	int old_w = in_out_rect.width;
	int old_h = in_out_rect.height;

	in_out_rect.x = in_out_rect.x - pad_left;
	if (in_out_rect.x < 0)
		in_out_rect.x = 0;
	pad_left = old_x - in_out_rect.x;

	in_out_rect.y = in_out_rect.y - pad_top;
	if (in_out_rect.y < 0)
		in_out_rect.y = 0;
	pad_top = old_y - in_out_rect.y;

	int tmp_w = old_x + in_out_rect.width + pad_right;
	if (tmp_w > max_width)
		in_out_rect.width = max_width - in_out_rect.x;
	else in_out_rect.width = in_out_rect.width + pad_right + pad_left;
	pad_right = in_out_rect.width - old_w - pad_left;

	int tmp_h = old_y + in_out_rect.height + pad_bottom;
	if (tmp_h > max_height)
		in_out_rect.height = max_height - in_out_rect.y;
	else in_out_rect.height = in_out_rect.height + pad_bottom + pad_top;
	pad_bottom = in_out_rect.width - old_w - pad_top;
}

inline void adjust_rect(TRect& in_out_rect, int std_width, int std_height, int patch_left, int patch_top, int patch_right, int patch_bottom)
{
    int left = in_out_rect.x+patch_left;
    int top = in_out_rect.y+patch_top;
    int right = in_out_rect.x+in_out_rect.width+patch_right;
    int bottom = in_out_rect.y+in_out_rect.height+patch_bottom;
    if(left<0)left = 0;
    if(top<0)top = 0;
    if(right>std_width)right=std_width;
    if(bottom>std_height)bottom=std_height;
    if(left<right && top<bottom)
    {
        in_out_rect.x = left;
        in_out_rect.y = top;
        in_out_rect.width = right-left;
        in_out_rect.height = bottom-top;
    }
}

template <typename DTYPE>
inline bool test_collision(const TemRect<DTYPE>& rect1, const TemRect<DTYPE>& rect2)
{
    int x1 = rect1.x;
    int y1 = rect1.y;
    int x2 = rect1.x + rect1.width;
    int y2 = rect1.y + rect1.height;

    int x3 = rect2.x;
    int y3 = rect2.y;
    int x4 = rect2.x + rect2.width;
    int y4 = rect2.y + rect2.height;

    return ( ( (x1 >=x3 && x1 < x4) || (x3 >= x1 && x3 <= x2) ) &&
        ( (y1 >=y3 && y1 < y4) || (y3 >= y1 && y3 <= y2) ) ) ? true : false;

}

#endif // COMMON_H
