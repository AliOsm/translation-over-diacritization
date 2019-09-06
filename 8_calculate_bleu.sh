echo "Without Diacritics BLEU Score"
perl mosesdecoder/scripts/generic/multi-bleu.perl data_dir/en.bpe.test < data_dir/ar.bpe.test.predictions.detok

echo "With Diacritics BLEU Score"
perl mosesdecoder/scripts/generic/multi-bleu.perl data_dir/en.bpe.test < data_dir/ar-diac.bpe.test.predictions.detok
