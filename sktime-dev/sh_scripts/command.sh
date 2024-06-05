pyversion=py310
base_image=python:3.10-slim
docker run -it \
    --privileged \
    -w /usr/src/sktime \
    --name sktime_${pyversion} \
    $base_image
# exit docker
ignore_list=(. .. .vscode .git env sktime-dev build htmlcov .pytest_cache testdir lightning_logs)
for f in $(ls -a $(pwd)); do
    if [[ " ${ignore_list[@]} " =~ [[:space:]]$f[[:space:]] ]]; then
        printf "ignore "
        printf $f
        printf "\n"
    else
        printf "copy "
        printf $f
        printf "\n"
        docker cp $(pwd)/$f sktime_${pyversion}:/usr/src/sktime/
    fi
done
# docker cp /home/xinyu/Documents/sktime sktime_py310:/usr/src
# enter docker
apt-get update && apt-get install --no-install-recommends -y build-essential gcc git && apt-get clean
python -m pip install -U pip
python -m pip install .[all_extras_pandas2,dev,binder]
python -m pip install pytorch-forecasting==1.0.0 pandas==2.2.1
python -m pip cache purge
rm -r /usr/src/sktime
# exit docker
docker commit sktime_${pyversion} sktime:${pyversion}


sktime-dev/sh_scripts/docker_test.sh -v ${pyversion} -m TestFull -f 1
