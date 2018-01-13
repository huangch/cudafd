#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#ifdef __cplusplus
extern "C" {
#endif

	int dbcCore(float * const result, int const * const image, int const height, int const width, int const s, int const G);
	int ebfdCore(float * const result, int const * const image, int const height, int const width, int const s, int const G);

#ifdef __cplusplus
}
#endif


#endif // __KERNEL_CUH__
