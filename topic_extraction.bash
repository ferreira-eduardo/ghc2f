python topic_extraction/topic_extraction.py --dataset_name amazon --dataset_path data --file_type parquet \
--text_col text --chunk_words 150 --use_cuml True --umap_n_jobs -1 --save_figures True \
--split_file splits/amazon_fold{fold}.csv --folds 0 1 2 3 4

python topic_extraction/topic_extraction.py --dataset_name imdb --dataset_path data --file_type parquet \
--text_col text --chunk_words 400 --use_cuml True --umap_n_jobs -1 --save_figures True \
--split_file splits/imdb_fold{fold}.csv --folds 0 1 2 3 4