if [ $# -eq 0 ]; then
    fuser -k /dev/nvidia*
else
    ps -x |grep $1 |awk '{print $1}' |xargs -I {} kill -9 {}
fi
