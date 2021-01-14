#ifndef CAFFE_PUBLIC_API
#define CAFFE_PUBLIC_API

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#define LIB_EXPORTS

#ifdef LIB_EXPORTS
#if defined(_MSC_VER)
#define LIB_API __declspec(dllexport)
#else
#define LIB_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define LIB_API
#else
#define LIB_API
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct detbox {
    int x, y, w, h;       // (x,y) - top-left corner, (w, h) - width & height of bounded box
    float prob;                    // confidence - probability that the object was found correctly
    int obj_id;           // class of object - from range [0, classes-1]
} detbox;


struct image_t {
	int h;                        // height
	int w;                        // width
	int c;                        // number of chanels (3 - for RGB)
	float *data;                  // pointer to the image data
};


// -----------------------------------------------------

LIB_API void free_detection_boxes(detbox *dets, int n);
LIB_API void free_ptr(void *ptr, int n);

LIB_API void caffe_recognize_image(char *cfgfile, char *weightfile, char *label_file, char *input_filename, char *output_text, int *num);
LIB_API void network_recognition_init(char *cfgfile, char *weightfile, char *label_file, char *device_name, int max_model_num);
//LIB_API void network_recognize_image(char *cfgfile, char *weightfile, char *label_file, image_t im, char *output_text, int *num);
LIB_API void network_recognize_image(image_t im, char *output_text, int *num);
LIB_API void network_recognize_image_n(image_t im, char **output_text, int *lengths, int N);

LIB_API void network_init();
LIB_API void network_release();

//LIB_API image_t copy_imaget_from_bytes(char *pdata, int w, int h, int c);
//LIB_API void free_imaget(image_t m);


#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // CAFFE_API
