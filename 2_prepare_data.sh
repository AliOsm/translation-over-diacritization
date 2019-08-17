paste -d "\t" data_dir/ar data_dir/en > data_dir/merged
shuf data_dir/merged > data_dir/shuf_merged
head -n 1000000 data_dir/shuf_merged > data_dir/merged
awk -F "\t" '{print $1}' data_dir/merged > data_dir/ar
awk -F "\t" '{print $2}' data_dir/merged > data_dir/en
rm data_dir/merged data_dir/shuf_merged

echo "Learn BPE"
subword-nmt learn-bpe -s 32000 < data_dir/ar > data_dir/ar.codes
subword-nmt learn-bpe -s 32000 < data_dir/en > data_dir/en.codes

echo "Apply BPE"
subword-nmt apply-bpe -c data_dir/ar.codes < data_dir/ar > data_dir/ar.seg
subword-nmt apply-bpe -c data_dir/en.codes < data_dir/en > data_dir/en.seg

echo "Extract Train and Test Data"
paste -d "\t" data_dir/ar.seg data_dir/en.seg > data_dir/merged
head -n 10000 data_dir/merged > data_dir/test
tail -n 990000 data_dir/merged > data_dir/train
rm data_dir/merged data_dir/ar.seg data_dir/en.seg data_dir/ar.codes data_dir/en.codes 

awk -F"\t" '{print $1}' data_dir/test > data_dir/ar.bpe.test
awk -F"\t" '{print $2}' data_dir/test > data_dir/en.bpe.test
awk -F"\t" '{print $1}' data_dir/train > data_dir/ar.bpe.train
awk -F"\t" '{print $2}' data_dir/train > data_dir/en.bpe.train
rm data_dir/train data_dir/test

head -n 10000 data_dir/ar > data_dir/ar.org.test
tail -n 990000 data_dir/ar > data_dir/ar.org.train
head -n 10000 data_dir/en > data_dir/en.org.test
tail -n 990000 data_dir/en > data_dir/en.org.train
rm data_dir/ar data_dir/en
