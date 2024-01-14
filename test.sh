startTime=`date "+%F %T"`
TIME=$(date  "+%Y%m%d_%H%M%S")

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT

source py38/setup.bash
NNODES=1
NGPUS=8

CONFIG=$1
OUTOUT=output/${CONFIG}
if [ ! -d $OUTOUT ]; then
    mkdir -p $OUTOUT
fi
CONFIG=${OUTOUT}/${CONFIG}.yaml

logdir=${OUTOUT}/log-test

# kill
./kill.sh run.py
./kill.sh train.py
sleep 5

# train
python -m paddle.distributed.launch \
    --log_dir ${logdir} \
    --nnodes ${NNODES} \
    --nproc_per_node $NGPUS \
    tools/test.py \
    --config $CONFIG \
    --save_dir $OUTOUT \
    --do_eval \
    --model $OUTOUT/latest.pdparams

# burn
./burning.sh

endTime=`date "+%F %T"`
startTimestamp=`date -d "$startTime" +%s`
endTimestamp=`date -d "$endTime" +%s`
deltaTimestamp=$[ $endTimestamp - $startTimestamp ]
deltaTime="$(($deltaTimestamp / 3600))h$((($deltaTimestamp % 3600) / 60))m"
echo "$startTime ---> $endTime" "Total: $deltaTime"
