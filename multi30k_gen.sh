DATE="2016"
CUDA_VISIBLE_DEVICES=3 python cd_gen.py --source_file /ltstorage/home/2pan/dataset/multi30k/data/task1/test/test_${DATE}_flickr.en --target_file /ltstorage/home/2pan/dataset/multi30k/data/task1/test/test_${DATE}_flickr.de --image_folder /ltstorage/home/2pan/dataset/Flickr/flickr30k-images/ --image_source /ltstorage/home/2pan/dataset/multi30k/data/task1/image_splits/test_${DATE}_flickr.txt --prompt_temp "Please translate the following English sentence into German:" --output_path evaluations/multi30k/no_am/mcd_no_temp/ 