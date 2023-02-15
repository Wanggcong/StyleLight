*Q1: C++11 is required to use dlib, but the version of GCC you are using is too old and doesn't support C++11.  You need GCC 4.9 or newer.
      Call Stack (most recent call first):
        CMakeLists.txt:8 (include)

*A1: check the gcc version in the terminal by: 

**1) gcc --version, e.g., /usr/local/bin/gcc

**2) set the path in ~/.bashrc : export CC=/usr/local/bin/gcc 

**3) update the environment by: source ~/.bashrc (twice)
