sensors=('AT305' 'AT203' 'FTBG') 

# sensors=('AT305 AT203' 'AT305 AT303' 'AT305 AT304' 'AT305 FTBG' 'AT203 AT201' 'AT203 AT202')
 
# sensors=('AT203 AT303 FTBG' 'AT203 AT305 FTBG' 'AT303 AT304 AT305')

# sensors=('AT303 AT304 AT305 FTBG')

num_layers_s=(1 2 3)

for sensor in ${sensors[*]}
do
	for num_layers in ${num_layers_s[*]}
	do
			echo -e "#!/bin/bash\n#SBATCH --job-name=hparam_search\n#SBATCH --output=logs/hparam_search.%j.out\n#SBATCH --time=5:00:00\n#SBATCH -p gpu\n#SBATCH -c 4\n#SBATCH --gres gpu:1\nmodule load python/3.6.1\nmodule load py-pandas/0.23.0_py36\nmodule load py-numpy/1.14.3_py36\nmodule load py-tensorflow/1.9.0_py36\ncd cs230/test_notebooks\npython3 seq_multichannel_hparam_search.py '$sensors' $num_layers" >| "temp.sbatch"
	done
done


