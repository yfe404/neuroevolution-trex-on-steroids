#!/bin/bash

echo "todo"
echo "parse ports.txt"

ROOT=$(pwd)

if [ -e pid.txt ]
then
    for pid in $(cat pid.txt); do kill $pid ;done
    echo "" > pid.txt
fi

while read src dst
do
    dir=$(mktemp -d)
    cp -r game/* $dir
    echo $dir
    cd $dir
    sed -Ei "s/127\.0\.0\.1:[[:digit:]]+/127\.0\.0\.1:$dst/g" ./index.js
    python -m http.server $src &
    _pid=$!
    echo "$_pid" >> $ROOT/pid.txt
    cd $ROOT
done < ports.txt


echo "create a copy of game directory in a mktemp "
echo "modufy the index.js file to listen to websocket on port dst"

