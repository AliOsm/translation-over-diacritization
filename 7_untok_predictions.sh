sed -r 's/(@@ )|(@@ ?$)//g' data_dir/ar.bpe.test.predictions > data_dir/ar.bpe.test.predictions.untok
sed -r 's/(@@ )|(@@ ?$)//g' data_dir/ar-diac.bpe.test.predictions > data_dir/ar-diac.bpe.test.predictions.untok
sed -i -r 's/ <end> //g' data_dir/ar.bpe.test.predictions.untok
sed -i -r 's/ <end> //g' data_dir/ar-diac.bpe.test.predictions.untok
