#!/bin/bash




for i in $(seq 1 3); do
    dir="$i_$(mktemp -d)"
    port=$((9921+$i))
    
    cp -R /tmp/neuroevolution-trex-on-steroids $dir
    cd $dir/neuroevolution-trex-on-steroids
    sed -Ei "s/127\.0\.0\.1:[[:digit:]]+/127\.0\.0\.1:$port/g" ./game/index.js
    source ./venv/bin/activate

    cd game
    python -m http.server $((8921+$i)) &
    cd ..
    
    python main.py $port &
    sleep 3
    firefox "http://localhost:$((8921+$i))" &
done


