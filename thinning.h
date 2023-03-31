#ifndef ACANE_THINNING_H_
#define ACANE_THINNING_H_

#include <cstdint>

#define ACANE_API __attribute__((visibility("default")))

#if __cplusplus
extern "C" {
#endif // __cplusplus

/// @brief Perform thining on a binary-value image
/// @param src  pointer to source binary image (value range: [0, 1])
/// @param dst  pointer to dest binary image (value range: [0, 1])
/// @param rows rows of the binary image
/// @param cols cols of the binary image
/// @param max_iteration
ACANE_API void thinning(const uint8_t* src, uint8_t* dst, int rows, int cols, int max_iteration);

#if __cplusplus
}
#endif // __cplusplus

#endif // ACANE_THINNING_H_
