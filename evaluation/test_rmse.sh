
# test_data_path='out_test6_no_coor'
test_data_path=$1
python fast_rmse.py --fake $test_data_path/mirror --real mirror
python fast_rmse.py --fake $test_data_path/matte_silver --real silver
python fast_rmse.py --fake $test_data_path/diffuse --real diffuse
