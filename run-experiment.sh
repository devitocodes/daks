VM_TYPE=$1
shift
EXPERIMENT_NAME=$1
shift

PARAMS="$@"

exit_on_error() {
    exit_code=$1
    last_command=${@:2}
    if [ $exit_code -ne 0 ]; then
        >&2 echo "\"${last_command}\" command failed with exit code ${exit_code}."
        exit $exit_code
    fi
}

RG_NAME=NAV_DAKS_$EXPERIMENT_NAME
CLUSTERNAME=$EXPERIMENT_NAME
DIRNAME=$EXPERIMENT_NAME
LOCATION=eastus
export DEVITO_LOGGING=ERROR

echo Creating resource group
az group create --name $RG_NAME --location $LOCATION
echo Provisioning AKS...
az aks create --resource-group $RG_NAME --name $CLUSTERNAME --node-count 15 --generate-ssh-keys --attach-acr devitoaks --node-vm-size $VM_TYPE

exit_on_error $?

az aks get-credentials --resource-group $RG_NAME --name $CLUSTERNAME

echo Deploying kubernetes configuration
kubectl apply --context $CLUSTERNAME -f kubernetes/dask-cluster.yaml

mkdir $DIRNAME

echo Waiting for service to come online
while [ 1 ]
do
    DASK_IP=`kubectl --context $CLUSTERNAME describe services devito-server|grep "LoadBalancer Ingress:"|grep -oE '((1?[0-9][0-9]?|2[0-4][0-9]|25[0-5])\.){3}(1?[0-9][0-9]?|2[0-4][0-9]|25[0-5])'`
    if [ -z "$DASK_IP" ]
    then
	sleep 2
    else
	echo Got Dask IP $DASK_IP
	break
    fi
done

echo Lets give the workers some time to start
while [ 1 ]
do
    STARTING=`kubectl --context $CLUSTERNAME get pods|grep ContainerCreating`
    if [ -z "$STARTING" ]
    then
	break
    else
	echo $STARTING
	sleep 2
    fi
done
sleep 30
docker-compose run -e DASK_SERVER_IP=$DASK_IP -e BLOB_ACCOUNT_NAME=$BLOB_ACCOUNT_NAME -e BLOB_ACCOUNT_KEY=$BLOB_ACCOUNT_KEY -w /app/daks daks /venv/bin/python -u fwi/run.py --results-dir $DIRNAME $PARAMS 2>&1 | tee $DIRNAME/output.txt

echo All done. Deleting resources

kubectl config delete-context $CLUSTERNAME

az aks delete --name $CLUSTERNAME --resource-group $RG_NAME --yes --no-wait

az group delete --name $RG_NAME --yes --no-wait
az ad app list -o table | grep $CLUSTERNAME|  awk '{print "az ad app delete --id " $1}'|sh
