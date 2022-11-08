import cv2
import os

# fols = ['dev_depth', 'dev_images']

# for fol in fols:
# 	vids = os.listdir(fol)
# 	for vid in vids:
# 		vid_path = os.path.join(fol, vid)

# 		tmp = vid.split('_')
# 		vid_name, vid_label = tmp[0], int(tmp[1])

# 		imgs = os.listdir(vid_path)
# 		for img in imgs:
# 			cur_img_path = os.path.join(vid_path, img)

# 			tmp = img.split('_')
# 			tmp[1] = str(1 - int(tmp[1]))
# 			new_img_path = os.path.join(vid_path, ''.join(tmp))

# 			os.rename(cur_img_path, new_img_path)

# 		new_vid_label = str(1 - vid_label)
# 		new_vid_path = os.path.join(fol, vid_name + '_' + new_vid_label)

# 		os.rename(vid_path, new_vid_path)

def hehe(folder, suffix):
	subs = os.listdir(folder)
	for sub in subs:
		sub_path = os.path.join(folder, sub)
		imgs = os.listdir(sub_path)

		for img in imgs:
			pre = img.split('_')[0]
			pre = [x for x in pre]
			pre = '_'.join(pre)

			new_img = pre + '_' + suffix

			os.rename(os.path.join(sub_path, img), os.path.join(sub_path, new_img))

hehe('dev_depth', 'depth1D.jpg')
hehe('dev_images', 'scene.jpg')