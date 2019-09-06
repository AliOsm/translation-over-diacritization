cd shakkelha
pip install -r requirements.txt
python predict.py -t rnn -n 3 -s big -a 10 -in ../data_dir/ar.org.train -out ../data_dir/ar-diac.org.train
python predict.py -t rnn -n 3 -s big -a 10 -in ../data_dir/ar.org.test -out ../data_dir/ar-diac.org.test
cd ..
pip install -r requirements.txt