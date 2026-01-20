cat eval.txt |grep mean_reward |awk '{print $2}'|tr '\n' '; '|tr '.' ','
cat eval.txt |grep std_reward |awk '{print $2}'|tr '\n' '; '|tr '.' ','
cat eval.txt |grep episodes: |awk '{print $2}'|tr '\n' '; '|tr '.' ','
