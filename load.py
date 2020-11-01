## train set location
train_hr_img_path = '/home/aistudio/srdata/DIV2K_train_HR' 
train_lr_img_path = '/home/aistudio/srdata/DIV2K_train_LR_bicubic/X4'

## test set location
valid_hr_img_path = '/home/aistudio/srdata/DIV2K_valid_HR'
valid_lr_img_path = '/home/aistudio/srdata/DIV2K_valid_LR_bicubic/X4'


# load im path to list
def load_file_list(im_path,im_format):
    return glob.glob(os.path.join(im_path, im_format))

# read im to list
def im_read(im_path):
    im_dataset = []
    for i in range(len(im_path)):
        path = im_path[i]
        # imread -- bgr
        im_data = cv2.imread(path)
        # change im channels ==> bgr to rgb
        img = cv2.cvtColor(im_data, cv2.COLOR_BGR2RGB)
        #print(im_data.shape)
        im_dataset.append(im_data)
    return im_dataset

# crop
def random_crop(im_set, image_size):
    crop_set = []
    for im in im_set:
        #print(im.shape)
        # Random generation x,y
        h, w, _ = im.shape
        y = random.randint(0, h-image_size)
        x = random.randint(0, w-image_size)
        # Random screenshot
        cropIm = im[(y):(y + image_size), (x):(x + image_size)]
        crop_set.append(cropIm)
    return crop_set

# resize im // change im channels
def im_resize(imgs, im_w, im_h, pattern='rgb'):
    resize_dataset = []
    for im in imgs:
        im = cv2.resize(im, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
        resize_dataset.append(im)
    resize_dataset = np.array(resize_dataset,dtype='float32')
    return resize_dataset

# data standardization
def standardized(imgs):
    imgs = np.array([a / 127.5 - 1 for a in imgs])
    return imgs