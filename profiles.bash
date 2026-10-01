#!/bin/bash


#python text_representation/process_reviews.py all_beauty --stage train --category "All Beauty"

python text_representation/process_reviews.py imdb --stage train --category "Movies and Cinema"

python text_representation/process_reviews.py musical_instruments --stage train --category "Musical Instruments and audio"

python text_representation/process_reviews.py tripadvisor --stage train --category "Hotels and accommodation "
