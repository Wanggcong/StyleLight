

# test_data_path='out_test6_no_coor'
test_data_path=$1 
echo $test_data_path
sh render_D.sh $test_data_path
sh render_S.sh $test_data_path
sh render_M.sh $test_data_path

