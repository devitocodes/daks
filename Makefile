overthrust_3D_initial_model.h5: 
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5

overthrust_3D_true_model_2D.h5: util/slicer.py overthrust_3D_true_model.h5
	python util/slicer.py --filename overthrust_3D_true_model.h5 --datakey m

overthrust_3D_true_model.h5:
	wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5

overthrust_3D_initial_model_2D.h5: util/slicer.py overthrust_3D_initial_model.h5
	python util/slicer.py --filename overthrust_3D_initial_model.h5 --datakey m0

shots.blob: fwi/generate_shot_data.py overthrust_3D_true_model_2D.h5.blob
	python fwi/generate_shot_data.py && touch shots.blob

fwi: shots.blob fwi/run.py overthrust_3D_initial_model_2D.h5.blob
	python fwi/run.py --max-iter 1

overthrust_3D_initial_model.h5.blob: overthrust_3D_initial_model.h5 util/uploader.py
	python util/uploader.py --filename overthrust_3D_initial_model.h5 --container models && touch overthrust_3D_initial_model.h5.blob

overthrust_3D_true_model_2D.h5.blob: overthrust_3D_true_model_2D.h5 util/uploader.py
	python util/uploader.py --filename overthrust_3D_true_model_2D.h5 --container models && touch overthrust_3D_true_model_2D.h5.blob

overthrust_3D_true_model.h5.blob: overthrust_3D_true_model.h5 util/uploader.py
	python util/uploader.py --filename overthrust_3D_true_model.h5 --container models && touch overthrust_3D_true_model.h5.blob

overthrust_3D_initial_model_2D.h5.blob: overthrust_3D_initial_model_2D.h5 util/uploader.py
	python util/uploader.py --filename overthrust_3D_initial_model_2D.h5 --container models && touch overthrust_3D_initial_model_2D.h5.blob

fwi_experiment: fwi_lossy

fwi_reference:
	python fwi/run.py --results-dir fwi_reference --nshots 80 --shots-container shots-rho-80-so-8 --so 8

fwi_lossy: fwi_reference

