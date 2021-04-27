#! /usr/bin/zsh

for F in $(ls /home/common/datasets/amazon_review_data_2018/reviews/); do
	zcat /home/common/datasets/amazon_review_data_2018/reviews/$F | jq '.["reviewText"]' | tr 'a-z' 'A-Z' | tr -sC 'A-Z' | tr -s ' ' '\n' | sort | uniq -c | sort -n > /home/nobr/bridger/data/counts/$F.out
	echo $F
done

