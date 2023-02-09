import os

# rename constants.py
os.system("mv ./constants.py ./temp_constants.py")

# fang
# 1.location
# os.system("cp ./blackbox_optimized_constant/blackbox_op_fang_location.py ./constants.py")
# os.system("python ./blackbox_optimized.py")
# os.system("rm ./constants.py")

# # median
# # 1.location
os.system("cp ./blackbox_optimized_constant/blackbox_op_median_location.py ./constants.py")
os.system("python ./blackbox_optimized.py")
os.system("rm ./constants.py")

# os.system("cp ./graybox_optimized_constant/graybox_op_fang_location.py ./constants.py")
# os.system("python ./graybox_optimized.py")
# os.system("rm ./constants.py")

# # median
# # 1.location
os.system("cp ./graybox_optimized_constant/graybox_op_median_location.py ./constants.py")
os.system("python ./graybox_optimized.py")
os.system("rm ./constants.py")

os.system("mv ./temp_constants.py ./constants.py")