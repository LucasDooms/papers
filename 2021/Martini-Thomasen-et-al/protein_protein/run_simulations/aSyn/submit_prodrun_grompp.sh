#!/bin/bash

t=283

for i in $(seq 1 10)
do

cd two_aSyn_$i



for j in 1.00 1.06 1.08
do

mkdir lambda_${j}
cd lambda_${j}

cp ../all_PRO_lambda${j}.top .
mv all_PRO_lambda${j}.top all_PRO_lambda.top

cp ../../prodrun_grompp.sh .
qsub prodrun_grompp.sh -v temp=$t
cd ..

done

cd ..

done

