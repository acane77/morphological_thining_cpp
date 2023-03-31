# Morphological Thinning

This repo is a C++ implementation of <a href="https://scikit-image.org/">scikit-image</a>'s <a href="https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.thin">skimage.morphology.thin</a>.

My implementation does not require any third-party dependencies. To use this project, just copy the header file and source file to your project.

The fundamental code structure is from <a href="https://github.com/bsdnoobz/zhang-suen-thinning">bsdnoobz/zhang-suen-thinning</a>, which performs thinning with Zhang-Suen's algorithm.
My implementation aligned with skimage's thinning with simple code, which produces indentical results to skimage's thinning.

This project is licensed under MIT license.

------

Relevant implementation in Python is `thining.py`. 
The python code compares my implementation(`thinning()`) with `skimage.morphology.thin()`.

While re-writing code from Python to C++, 
you can first replace the `thin` function in python to make sure the results are still correct,
then use the C++ version in your related C++ code.

------

Sample output:
```
Before thinning:
................................
................................
................................
................................
..........#########.............
..........#############.........
................................
................................
................................
................................
.....#####################......
.....#####################......
................................
................................
................................
................................

After thinning:
................................
................................
................................
................................
................................
..........#############.........
................................
................................
................................
................................
................................
.....####################.......
................................
................................
................................
................................

```