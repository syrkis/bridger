#! /usr/bin/zsh

for F in $(ls /home/common/datasets/amazon_review_data_2018/reviews/); do
	if [ ! -f "/home/nobr/bridger/data/counts/$(echo $F | cut -d '.' -f 1).out" ]; then
		zcat /home/common/datasets/amazon_review_data_2018/reviews/$F | jq '.["reviewText"]' | tr 'a-z' 'A-Z' | tr -sC 'A-Z' ' ' | tr -s ' ' '\n' | sort | uniq -c | sort -n > /home/nobr/bridger/data/counts/$(echo $F | cut -d '.' -f 1).out
		echo "JUST DID $F"
	else
		echo "ALREADY DONE $F"	
	fi
done

echo "GOD BLESS THE MACHINE"
