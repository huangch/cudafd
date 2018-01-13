#ifndef __cudaFD_H__
#define __cudaFD_H__

#ifdef __cplusplus
extern "C" {
#endif
	int dbc(float * const result, int const * const image, int const height, int const width, int const s, int const G);
	int ebfd(float * const result, int const * const image, int const height, int const width, int const s, int const G);
#ifdef __cplusplus
}
#endif

#endif // __cudaFD_H__
