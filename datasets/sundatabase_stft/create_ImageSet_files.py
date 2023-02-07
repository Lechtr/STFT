import os
import random

dest_path = "D:\\Master_Daten\\sunddatabase_stft\\ImageSets"
# used for counting the images
annotations_path = "D:\\Master_Daten\\sunddatabase_stft\\XML_adenoma_vs_non_adenoma"

cases = list(range(1, 101))

random.shuffle(cases)
cases_val = cases[:10]
cases_train = cases[10:]


# cases_val = [4, 10]
# cases_train = [36, 37, 38, 39]

print("Validation")
with open(os.path.join(dest_path, 'sundatabase_stft_val_videos.txt'), 'w') as f:
    for case in cases_val:
        case_dir = os.path.join(annotations_path, "case" + str(case))
        max_image = len([name for name in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, name))])
        print(max_image)

        for img_idx in range(1, max_image+1):
            f.write("case" + str(case) + " " + str(img_idx) + " " + str(img_idx) + " " + str(max_image))
            f.write('\n')


print("Training")
with open(os.path.join(dest_path, 'sundatabase_stft_train_videos.txt'), 'w') as f:
    for case in cases_train:
        case_dir = os.path.join(annotations_path, "case" + str(case))
        max_image = len([name for name in os.listdir(case_dir) if os.path.isfile(os.path.join(case_dir, name))])
        print(max_image)

        for img_idx in range(1, max_image+1):
            f.write("case" + str(case) + " " + str(img_idx) + " " + str(img_idx) + " " + str(max_image))
            f.write('\n')




