
# test_data_path='out_test6_no_coor_exr'
# out_dir='out_test6_no_coor'

out_dir=$1
test_data_path=$2   # where is  predicted .exr
# python tonemap.py --out_dir $out_dir --testdata ./data/original/$test_data_path/ 
echo $out_dir
echo $test_data_path
python tonemap.py --out_dir $out_dir --testdata $test_data_path #./data/original/$test_data_path/ 