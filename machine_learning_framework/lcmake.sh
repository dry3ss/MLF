#!/bin/sh

cd "build_l"
cmake .. -G "Sublime Text 2 - Unix Makefiles"
make
cd ".."
