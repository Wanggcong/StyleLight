#!/bin/dash
# alias blender=/home/deep/Downloads/blender-2.93.4-linux-x64/blender
alias blender=/home/deep/Downloads/blender-3.0.1-linux-x64/blender

test_data_path=$1

#conda list
blender --background --python mirror.py -- 50 5 $test_data_path 0 10
blender --background --python mirror.py -- 50 5 $test_data_path 10 20
blender --background --python mirror.py -- 50 5 $test_data_path 20 30
blender --background --python mirror.py -- 50 5 $test_data_path 30 40
blender --background --python mirror.py -- 50 5 $test_data_path 40 50
blender --background --python mirror.py -- 50 5 $test_data_path 50 60
blender --background --python mirror.py -- 50 5 $test_data_path 60 70
blender --background --python mirror.py -- 50 5 $test_data_path 70 80
blender --background --python mirror.py -- 50 5 $test_data_path 80 90
blender --background --python mirror.py -- 50 5 $test_data_path 90 100
blender --background --python mirror.py -- 50 5 $test_data_path 100 110
blender --background --python mirror.py -- 50 5 $test_data_path 110 120
blender --background --python mirror.py -- 50 5 $test_data_path 120 130
blender --background --python mirror.py -- 50 5 $test_data_path 130 140
blender --background --python mirror.py -- 50 5 $test_data_path 140 150
blender --background --python mirror.py -- 50 5 $test_data_path 150 160
blender --background --python mirror.py -- 50 5 $test_data_path 160 170
blender --background --python mirror.py -- 50 5 $test_data_path 170 180
blender --background --python mirror.py -- 50 5 $test_data_path 180 190
blender --background --python mirror.py -- 50 5 $test_data_path 190 200
blender --background --python mirror.py -- 50 5 $test_data_path 200 210
blender --background --python mirror.py -- 50 5 $test_data_path 210 220
blender --background --python mirror.py -- 50 5 $test_data_path 220 230
blender --background --python mirror.py -- 50 5 $test_data_path 230 240
blender --background --python mirror.py -- 50 5 $test_data_path 240 250
blender --background --python mirror.py -- 50 5 $test_data_path 250 260
blender --background --python mirror.py -- 50 5 $test_data_path 260 270
blender --background --python mirror.py -- 50 5 $test_data_path 270 280
blender --background --python mirror.py -- 50 5 $test_data_path 280 289



