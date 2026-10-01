#!/bin/bash

data_path='../dataset/'
#
#uv run python aspects/aspects_extraction_absa_pt.py --dataset_path=$data_path --dataset_name amazon \
#--is_sample True --file_type parquet --user_col userId --item_col itemId --text_col review \
#
#uv run python aspects/aspects_extraction_absa_pt.py --dataset_path=$data_path --dataset_name imdb \
#--is_sample True --file_type parquet --user_col userId --item_col itemId --text_col review
#
#uv run python aspects/aspects_extraction_absa_pt.py --dataset_path=$data_path --dataset_name tripadvisor \
#--is_sample True --file_type parquet --user_col userId --item_col itemId --text_col review

#uv run python aspects/aspects_extraction_absa_pt.py --dataset_path=$data_path --dataset_name Musical_Instruments_reviews \
#--is_sample True --file_type jsonl --user_col user_id --item_col parent_asin --text_col text

uv run python aspects/aspects_extraction_absa_pt.py --dataset_path="$data_path" --dataset_name all_beauty \
  --file_type parquet --user_col userId --item_col itemId --text_col text \
  --is_sample False --context_window 1 --full_review_max_chars 0 \
  --min_reviews 1 --min_items 1