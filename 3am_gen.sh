CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=3 python llava_gen.py --source_file /ltstorage/home/2pan/dataset/3AM/data/test.en --target_file /ltstorage/home/2pan/dataset/3AM/data/test.zh --image_folder /ltstorage/home/2pan/dataset/3AM/images/ --image_source /ltstorage/home/2pan/dataset/3AM/data/images-test.txt --prompt_temp "Please translate the following English sentence into Chinese:" --output_path evaluations/3am/only_text/