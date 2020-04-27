
#!/bin/bash

for RangePow in {1..18}
do
   ./runme ../SLIDE/Config_amz.csv $RangePow &> sparse.$RangePow &
done

wait


