#!/bin/bash

for pid in $(cat pid.txt) ; do kill -9 $pid ;done
