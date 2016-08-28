# Desert Deep Dream
Interactive DeepDream, as seen at Burning Man.

http://desertdeepdreams.wordpress.com/

[![Slugdog playa](http://i.imgur.com/TtaIWRPl.jpg)](http://i.imgur.com/TtaIWRP.jpg)
[![Spiral people](http://i.imgur.com/TFq0iFAl.jpg)](http://i.imgur.com/TFq0iFA.jpg)

## About

Interactive DeepDream installation. Takes a picture, and displays it iteratively made trippier.

Background:
* https://en.wikipedia.org/wiki/DeepDream
* https://research.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html

This code is not pretty; a significant amount of it was written on the playa in 2015 and barely touched since. Any enhancements made
in 2016 will be pushed to this repository; I make no promises about cleanup.

## Dependencies

* NumPy/SciPy
* Caffe
* PIL
* OpenCV (used only for camera access and display)

On Windows (what this has been used on in production), the directions at
http://thirdeyesqueegee.com/deepdream/2015/07/19/running-googles-deep-dream-on-windows-with-or-without-cuda-the-easy-way/
will install all of these dependencies except OpenCV.

Anaconda's OpenCV doesn't have camera support; installing one with camera support
involves:

* Downloading from http://opencv.org/downloads.html (3.0 is the version tested with)
* Copying `build\python\2.7\x64\cv2.pyd` to `Lib\site-packages` in the Anaconda directory
* Copying `sources\3rdparty\ffmpeg\opencv_ffmpeg_64.dll` to *a file named `opencv_ffmpegNNN_64.dll`*, where NNN is the 3-digit OpenCV version with dots removed (e.g. 300 for 3.0) in the root of the Anaconda directory

You'll also need neural-network models. The two models referred to in the source (BVLC GoogLeNet from the Berkeley Vision and Learning
Center (aka "slugdog"), and Places205-GoogLeNet from MIT) are available at http://nickptar.s3.amazonaws.com/deepdream-models.zip. Extract
that to the directory containing the Python files.

## Running
cd to the directory containing the Python files and run `python capdream.py`.

Two windows will open. The intended hardware platform has a control display (e.g. laptop screen) and a public display connected
to the computer and configured as separate monitors.
The "Inside" window is intended to stay on the control display, and the "Outside" window is intended to be moved to the public display.
The "Outside" window will toggle between windowed and fullscreen when the F key is pressed over either window.
Press Q over either window to quit.

Final images will be saved in the `saves` directory.

A lot can be configured in the TWEAKABLES section at the top of capdream.py:

* Which layers of which neural networks to use (the layer to use for an image is randomly chosen from the options, without repeating the
  same one successively).
* How long and how to iterate:
  * octaves is the number of zoom levels
  * iter_n is the number of iterations per octave (so the total number of iterations is `iter_n * octave`; iterations at later octaves are exponentially slower)
  * oct_scale is the zoom ratio between zoom levels
* process_h sets the height of the image while it's being processed, and render_h the height when it's displayed. The width is determined
  by the aspect ratio of the camera, but init_render_w is the width used to initialize the windows.
* camera_idx is the camera index to use, if you have multiple cameras.
* flash_intensity is the brightness of the flash, which is used to take a better photo during dark hours ("dark hours" are
  currently determined in the `snap` function, and currently range from 20:00 to 6:00, approximating twilight/night in northwest
  Nevada in the fall)
* Other tweakables will hopefully be self-explanatory.

## Credit
The only code that's mine is the front end in capdream.py. The interesting parts of dreamify.py are by the original team at Google, and
credit for what's further behind that goes to Google, UC Berkeley (Caffe), and many others.

In-person thanks for supporting this and helping me iterate on it go to Elizabeth Tarleton, Robby Bensinger, Will and Divia Eden, and others I've forgotten.
