#!/bin/sh

c_DIR="/media/will/DonneesW/My_Neural_Network_framework/MLF/machine_learning_framework/"
cd "$c_DIR/build_l"
cmake .. -G "Sublime Text 2 - Unix Makefiles"
make
cd ".."
