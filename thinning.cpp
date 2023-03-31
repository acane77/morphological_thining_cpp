#include "thinning.h"

#include <vector>
#include <cassert>
#include <cstring>
#include <cstdint>

namespace {

    const static uint8_t G123_LUT[256] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1,
            0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
            0, 1, 1, 0, 0, 1, 0, 0, 0
    };

    const static uint8_t G123P_LUT[256] = {
            0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
            0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0
    };

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * Parameters:
 * 		im    Binary image with range = [0,1]
 * 		iter  0=even, 1=odd
 */
    void thinning_iteration(uint8_t *img, const uint8_t* lud, int rows, int cols) {
        assert(rows > 3 && cols > 3);

        int img_size = rows * cols;

        std::vector<uint8_t> marker(img_size, 0);

        int x, y;
        uint8_t *pAbove;
        uint8_t *pCurr;
        uint8_t *pBelow;
        uint8_t *nw, *no, *ne;    // north (pAbove)
        uint8_t *we, *me, *ea;
        uint8_t *sw, *so, *se;    // south (pBelow)

        uint8_t *pDst;

#define GET_ROW(data, row) ((data) + (row) * cols)
        // initialize row pointers
        pAbove = nullptr;
        pCurr = GET_ROW(img, 0);
        pBelow = GET_ROW(img, 1);

        for (y = 1; y < rows - 1; ++y) {
            // shift the rows up by one
            pAbove = pCurr;
            pCurr = pBelow;
            pBelow = GET_ROW(img, y + 1);

            pDst = GET_ROW(marker.data(), y);

            // initialize col pointers
            no = &(pAbove[0]);
            ne = &(pAbove[1]);
            me = &(pCurr[0]);
            ea = &(pCurr[1]);
            so = &(pBelow[0]);
            se = &(pBelow[1]);

            for (x = 1; x < cols - 1; ++x) {
                // shift col pointers left by one (scan left to right)
                nw = no;
                no = ne;
                ne = &(pAbove[x + 1]);
                we = me;
                me = ea;
                ea = &(pCurr[x + 1]);
                sw = so;
                so = se;
                se = &(pBelow[x + 1]);

                pDst[x] = *ea + (*ne << 1) + (*no << 2) + (*nw << 3) + (*we << 4) +
                          (*sw << 5) + (*so << 6) + (*se << 7);
            }
        }

        for (int i = 0; i < rows * cols; i++) {
            if (lud[marker[i]])
                img[i] = 0;
        }
    }

    bool is_identical(const uint8_t *a, const uint8_t *b, int rows, int cols) {
        for (int i = 0; i < rows * cols; i++) {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }

}

void thinning(const uint8_t* src, uint8_t* dst, int rows, int cols, int max_iteration) {
    for (int i=0; i<rows * cols; i++) {
        dst[i] = src[i];// / 255.f; // convert to binary image
    }

    std::vector<uint8_t> prev(rows*cols, 0);
    bool diff;

    int num_iter = 0;
    do {
        thinning_iteration(dst, G123_LUT, rows, cols);
        thinning_iteration(dst, G123P_LUT, rows, cols);
        diff = !is_identical(dst, prev.data(), rows, cols);
        memcpy(prev.data(), dst, sizeof(uint8_t) * rows * cols);
        num_iter++;
    } while (diff && num_iter <= max_iteration);

//  for (int i=0; i<rows*cols; i++)
//    dst[i] *= 255;
}
