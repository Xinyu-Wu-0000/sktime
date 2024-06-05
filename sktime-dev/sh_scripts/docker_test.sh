sktime_dir=$(pwd)
pyversion="py310"
message="test"
test_mode=0
while getopts 'v:m:t:' opt; do
    case "${opt}" in
    v) pyversion="$OPTARG" ;;
    m) message="$OPTARG" ;;
    t) test_mode=$OPTARG ;;
    *)
        printf "error while parsing parameters\n"
        printf "v: python version\n"
        printf "m: test message\n"
        printf "t: test mode\n"
        exit 1
        ;;
    esac
done

if [ $test_mode == 0 ]; then
    test_script="test_full.sh"
    echo "full test"
elif [ $test_mode == 1 ]; then
    test_script="test_without_datasets.sh"
    echo "test without datasets"
elif [ $test_mode == 2 ]; then
    test_script="test_all_forecaster.sh"
    echo "test all forecaster"
else
    test_script="test_check_estimators.sh"
    echo "test check estimators"
fi

docker run -it --name sktime_${message}_${pyversion} --privileged=true sktime:${pyversion} bash -c "chmod +x ./$test_script && ./$test_script"
chmod +x sktime-dev/sh_scripts/*
ignore_list=(. .. .vscode .git env sktime-dev build htmlcov .pytest_cache testdir lightning_logs)
for f in $(ls -a $sktime_dir); do
    if [[ " ${ignore_list[@]} " =~ [[:space:]]$f[[:space:]] ]]; then
        printf "ignore "
        printf $f
        printf "\n"
    else
        printf "copy "
        printf $f
        printf "\n"
        docker cp $sktime_dir/$f sktime_${message}_${pyversion}:/usr/src/sktime/
    fi
done
docker cp sktime-dev/sh_scripts/$test_script sktime_${message}_${pyversion}:/usr/src/sktime/
docker cp sktime-dev/py_scripts/check_estimators_test.py sktime_${message}_${pyversion}:/usr/src/sktime/
# docker cp $sktime_dir sktime_${message}_${pyversion}:/usr/src
time stdbuf -oL docker start -i sktime_${message}_${pyversion} >testdir/results_log/${message}_${pyversion}.log 2>&1
