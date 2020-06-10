### DAKS - Azure Devito at scale using k8s and dask
For seismic imaging, the repository shows how you can leverage open-source PDE solver [Devito](https://www.devitoproject.org/), to perform Full-Waveform Inversion (FWI) at scale on Azure using Dask and k8s  

Steps to run:
1. Build the docker image: `docker-compose build` should create the docker image. 
2. Tag and upload the built image to ACR - note the name of the image. 
3. Update `dask-cluster.yaml` to use this docker image for both the scheduler and worker. 
4. Do the background to provision a kubernetes cluster and have `kubectl` working on it. 
5. `kubectl apply -f dask-cluster.yaml`
6. Get the dask scheduler IP from this cluster and put it in `docker-compose.yaml`
7. Put your azure storage credentials in `azureio.py`
6. `docker-compose run daks /bin/bash`
7. `cd daks`
8. `make shots.blob` - this downloads the 3D Overthrust Model from SLIM group's FTP server (thanks to them for providing this) to the local directory, extracts a 2D slice, uploads the 2D slice to blob storage, and then runs the shot generation script that reads from blob storage directly on the k8s cluster and writes shot data to blob storage directly. 
9. You should now have blobs named `shot_0.h5` to `shot_19.h5` (for default nshots=20, can be passed as a command-line parameter to `python generate_shot_data.py --nshots <NUMBER>`) in a container called `shots` in your blob storage account. 
