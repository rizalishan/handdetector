# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.0

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone

# Include any dependencies generated for this target.
include CMakeFiles/Examples.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Examples.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Examples.dir/flags.make

CMakeFiles/Examples.dir/main.cpp.o: CMakeFiles/Examples.dir/flags.make
CMakeFiles/Examples.dir/main.cpp.o: main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/Examples.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/Examples.dir/main.cpp.o -c /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone/main.cpp

CMakeFiles/Examples.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Examples.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone/main.cpp > CMakeFiles/Examples.dir/main.cpp.i

CMakeFiles/Examples.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Examples.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone/main.cpp -o CMakeFiles/Examples.dir/main.cpp.s

CMakeFiles/Examples.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/Examples.dir/main.cpp.o.requires

CMakeFiles/Examples.dir/main.cpp.o.provides: CMakeFiles/Examples.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/Examples.dir/build.make CMakeFiles/Examples.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/Examples.dir/main.cpp.o.provides

CMakeFiles/Examples.dir/main.cpp.o.provides.build: CMakeFiles/Examples.dir/main.cpp.o

# Object files for target Examples
Examples_OBJECTS = \
"CMakeFiles/Examples.dir/main.cpp.o"

# External object files for target Examples
Examples_EXTERNAL_OBJECTS =

Examples: CMakeFiles/Examples.dir/main.cpp.o
Examples: CMakeFiles/Examples.dir/build.make
Examples: /opt/X11/lib/libX11.dylib
Examples: /usr/local/lib/libopencv_core.a
Examples: /usr/local/lib/libopencv_flann.a
Examples: /usr/local/lib/libopencv_imgproc.a
Examples: /usr/local/lib/libopencv_highgui.a
Examples: /usr/local/lib/libopencv_features2d.a
Examples: /usr/local/lib/libopencv_calib3d.a
Examples: /usr/local/lib/libopencv_ml.a
Examples: /usr/local/lib/libopencv_video.a
Examples: /usr/local/lib/libopencv_legacy.a
Examples: /usr/local/lib/libopencv_objdetect.a
Examples: /usr/local/lib/libopencv_photo.a
Examples: /usr/local/lib/libopencv_gpu.a
Examples: /usr/local/lib/libopencv_videostab.a
Examples: /usr/local/lib/libopencv_ts.a
Examples: /usr/local/lib/libopencv_ocl.a
Examples: /usr/local/lib/libopencv_superres.a
Examples: /usr/local/lib/libopencv_nonfree.a
Examples: /usr/local/lib/libopencv_stitching.a
Examples: /usr/local/lib/libopencv_contrib.a
Examples: /usr/local/lib/libopencv_nonfree.a
Examples: /usr/local/lib/libopencv_gpu.a
Examples: /usr/local/lib/libopencv_legacy.a
Examples: /usr/local/lib/libopencv_photo.a
Examples: /usr/local/lib/libopencv_ocl.a
Examples: /usr/local/lib/libopencv_calib3d.a
Examples: /usr/local/lib/libopencv_features2d.a
Examples: /usr/local/lib/libopencv_flann.a
Examples: /usr/local/lib/libopencv_ml.a
Examples: /usr/local/lib/libopencv_video.a
Examples: /usr/local/lib/libopencv_objdetect.a
Examples: /usr/local/lib/libopencv_highgui.a
Examples: /usr/local/lib/libopencv_imgproc.a
Examples: /usr/local/lib/libopencv_core.a
Examples: /usr/local/share/OpenCV/3rdparty/lib/liblibjpeg.a
Examples: /usr/local/share/OpenCV/3rdparty/lib/liblibpng.a
Examples: /usr/local/share/OpenCV/3rdparty/lib/liblibtiff.a
Examples: /usr/local/share/OpenCV/3rdparty/lib/liblibjasper.a
Examples: /usr/local/share/OpenCV/3rdparty/lib/libIlmImf.a
Examples: /usr/local/share/OpenCV/3rdparty/lib/libzlib.a
Examples: CMakeFiles/Examples.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable Examples"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Examples.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Examples.dir/build: Examples
.PHONY : CMakeFiles/Examples.dir/build

CMakeFiles/Examples.dir/requires: CMakeFiles/Examples.dir/main.cpp.o.requires
.PHONY : CMakeFiles/Examples.dir/requires

CMakeFiles/Examples.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Examples.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Examples.dir/clean

CMakeFiles/Examples.dir/depend:
	cd /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone /Users/rizwan/Documents/RWTH/S2/VR/OPENCV/SkinTone/CMakeFiles/Examples.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Examples.dir/depend

