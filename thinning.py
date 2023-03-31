from skimage.morphology._skeletonize import thin  # for compare
import numpy as np

G123_LUT = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
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
                     0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)

G123P_LUT = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
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
                      0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)


def _thinningIteration(im, lut):
    # neighborhood mask
    w = np.array([[8, 4, 2],
                  [16, 0, 1],
                  [32, 64, 128]], dtype=np.uint8)

    I, M = im, np.zeros(im.shape, np.uint8)
    for i in range(1, im.shape[0]-1):
        for j in range(1, im.shape[1]-1):
            p2 = im[i-1, j]
            p2w = w[0, 1]
            p3 = im[i-1, j+1]
            p3w = w[0, 2]
            p4 = im[i, j+1]
            p4w = w[1, 2]
            p5 = im[i+1, j+1]
            p5w = w[2, 2]
            p6 = im[i+1, j]
            p6w = w[2, 1]
            p7 = im[i+1, j-1]
            p7w = w[2, 0]
            p8 = im[i, j-1]
            p8w = w[1, 0]
            p9 = im[i-1, j-1]
            p9w = w[0, 0]
            M[i, j] = p2 * p2w + p3 * p3w + p4 * p4w + p5 * p5w + p6 * p6w + \
                      p7 * p7w + p8 * p8w + p9 * p9w
    D = np.take(lut, M)
    im[D] = 0

    return im

def thinning(src):
    dst = src.copy()
    prev = np.zeros(src.shape[:2], np.uint8)
    diff = None

    while True:
        dst = _thinningIteration(dst, G123_LUT).copy()
        dst = _thinningIteration(dst, G123P_LUT).copy()
        diff = np.absolute(dst - prev)
        prev = dst.copy()
        if np.sum(diff) == 0:
            break

    return dst



def main():
    input_img = np.load('input.npy')
    out_img1 = thin(input_img.astype(np.uint8))
    input_img = np.load('input.npy')
    out_img2 = thinning(input_img.astype(np.uint8))
    cmp =(out_img1 == out_img2).astype(np.uint8).sum()
    identical = cmp == input_img.shape[0] * input_img.shape[1]
    if (identical):
        print("identical  -- aligned")
    else:
        print("not identical  -- not aligned")
        print(out_img1 == out_img2)

if __name__ == '__main__':
    main()