sed -r 's/(@@ )|(@@ ?$)//g' data_dir/ar.bpe.test.predictions > data_dir/ar.bpe.test.predictions.detok
sed -r 's/(@@ )|(@@ ?$)//g' data_dir/ar-diac.bpe.test.predictions > data_dir/ar-diac.bpe.test.predictions.detok
sed -i -r 's/ <end> //g' data_dir/ar.bpe.test.predictions.detok
sed -i -r 's/ <end> //g' data_dir/ar-diac.bpe.test.predictions.detok

echo "Finished detokenizing!"