Require
-------
- CNTK version 2.0 Beta 12
  https://github.com/Microsoft/CNTK/releases
- NVIDIA GPU


Additional Options
------------------
--no-nn            Do not use neural networks.

--no-gpu           Do not use GPU, even if GPU is available.

--no-early-pass    Do not pass.
                   (for CGOS)

--device-id 0      Set GPU to use.
                   --device-id X where X is an integer >=0 means use GPU X, i.e. deviceId=0 means GPU 0, etc.
