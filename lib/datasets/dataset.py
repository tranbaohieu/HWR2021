from __future__ import absolute_import

# import sys
# sys.path.append('./')

import os
# import moxing as mox

import pickle
from tqdm import tqdm
from PIL import Image, ImageFile
import numpy as np
import random
import cv2
import lmdb
import sys
import six
import math

import torch
from torch.utils import data
from torch.utils.data import sampler
from torchvision import transforms

from lib.utils.labelmaps import get_vocabulary, labels2strs
from lib.utils import to_numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np


from config import get_args
global_args = get_args(sys.argv[1:])

if global_args.run_on_remote:
  import moxing as mox

class LmdbDataset(data.Dataset):
  def __init__(self, root, voc_type, max_len, num_samples, transform=None):
    super(LmdbDataset, self).__init__()

    if global_args.run_on_remote:
      dataset_name = os.path.basename(root)
      data_cache_url = "/cache/%s" % dataset_name
      if not os.path.exists(data_cache_url):
        os.makedirs(data_cache_url)
      if mox.file.exists(root):
        mox.file.copy_parallel(root, data_cache_url)
      else:
        raise ValueError("%s not exists!" % root)
      
      self.env = lmdb.open(data_cache_url, max_readers=32, readonly=True)
    else:
      self.env = lmdb.open(root, max_readers=32, readonly=True)

    assert self.env is not None, "cannot create lmdb from %s" % root
    self.txn = self.env.begin()

    self.voc_type = voc_type
    self.transform = transform
    self.max_len = max_len
    self.nSamples = int(self.txn.get(b"num-samples"))
    self.nSamples = min(self.nSamples, num_samples)

    assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    self.EOS = 'EOS'
    self.PADDING = 'PADDING'
    self.UNKNOWN = 'UNKNOWN'
    self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    self.char2id = dict(zip(self.voc, range(len(self.voc))))
    self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(self.voc)
    self.lowercase = (voc_type == 'LOWERCASE')

  def __len__(self):
    return self.nSamples

  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
      # img = Image.open(buf).convert('L')
      # img = img.convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]

    # reconition labels
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.lowercase:
      word = word.lower()
    ## fill with the padding token
    label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
    label_list = []
    for char in word:
      if char in self.char2id:
        label_list.append(self.char2id[char])
      else:
        ## add the unknown token
        print('{0} is out of vocabulary.'.format(char))
        label_list.append(self.char2id[self.UNKNOWN])
    ## add a stop token
    label_list = label_list + [self.char2id[self.EOS]]
    assert len(label_list) <= self.max_len
    label[:len(label_list)] = np.array(label_list)

    if len(label) <= 0:
      return self[index + 1]

    # label length
    label_len = len(label_list)

    if self.transform is not None:
      img = self.transform(img)
    return img, label, label_len


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = np.asarray(img)/255
    img = img.astype(np.float32)
    # print(img.size)
    # img = self.toTensor(img)
    img = torch.from_numpy(img)
    img.sub_(0.5).div_(0.5)
    img = img.transpose(1, 0)
    img = img.transpose(2, 0)
    # img = img.to(dtype=torch.float64)
    # print(type(img))
    return img


class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())


class AlignCollate_padding(object):
    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
      self.imgH = imgH
      self.imgW = imgW
      self.keep_ratio = keep_ratio
      self.min_ratio = min_ratio

    def padding_mask(self, ii, aa1, bb1):
      ii = ii.float()
      img_size_h = ii.size()[1]
      img_size_w = ii.size()[2]
      # img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
      # img_mask_sub_s = img_mask_sub_s*255.0
      # img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
      padding_h = aa1-img_size_h
      padding_w = bb1-img_size_w
      m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
      img_mask_sub_padding = m(ii)
      img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
      return img_mask_sub_padding

    def resize(self, w, h, expected_height, image_min_width, image_max_width):
      new_w = int(expected_height * float(w) / float(h))
      round_to = 10
      new_w = math.ceil(new_w/round_to)*round_to
      new_w = max(new_w, image_min_width)
      new_w = min(new_w, image_max_width)

      return new_w, expected_height

    def process_image(self, image, image_height, image_min_width, image_max_width):
      img = image.convert('RGB')

      w, h = img.size
      new_w, image_height = self.resize(w, h, image_height, image_min_width, image_max_width)

      img = img.resize((new_w, image_height), Image.ANTIALIAS)

      img = np.asarray(img).transpose(2,0, 1)
      img = img/255
    

      return img

    def __call__(self, batch):
      images, labels, lengths = zip(*batch)
      b_lengths = torch.IntTensor(lengths)
      b_labels = torch.IntTensor(labels)

      imgH = self.imgH
      imgW = 0
      list_img = []
      for i,image in enumerate(images):
        image = self.process_image(image, imgH, 32, self.imgW)
        image = torch.FloatTensor(image)
        list_img.append(image)
        w = image.size()[2]
        if w > imgW:
            imgW = w
      imgW = self.imgW
      img_paddinges = []
      for i, image in enumerate(list_img):
        img = self.padding_mask(image, imgH, imgW)
        img = img.squeeze(0)
        # print(img.size())
        img_paddinges.append(img) 

      b_images = torch.stack(img_paddinges)
      print(b_images.size())


      return b_images, b_labels, b_lengths

class AlignCollate(object):

  def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    self.imgH = imgH
    self.imgW = imgW
    self.keep_ratio = keep_ratio
    self.min_ratio = min_ratio

  def __call__(self, batch):
    images, labels, lengths = zip(*batch)
    b_lengths = torch.IntTensor(lengths)
    b_labels = torch.IntTensor(labels)

    imgH = self.imgH
    imgW = self.imgW
    if self.keep_ratio:
      ratios = []
      for image in images:
        w, h = image.size
        ratios.append(w / float(h))
      ratios.sort()
      max_ratio = ratios[-1]
      imgW = int(np.floor(max_ratio * imgH))
      imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
      imgW = min(imgW, 400)

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)
    b_images = b_images.to(dtype=torch.float64)
    # b_images = b_images.transpose(3,1)
    print(b_images.size())

    return b_images, b_labels, b_lengths


def test():
  # lmdb_path = "/share/zhui/reg_dataset/NIPS2014"
  lmdb_path = "/share/zhui/reg_dataset/IIIT5K_3000"
  train_dataset = LmdbDataset(root=lmdb_path, voc_type='ALLCASES_SYMBOLS', max_len=50)
  batch_size = 1
  train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=AlignCollate(imgH=64, imgW=256, keep_ratio=False))

  for i, (images, labels, label_lens) in enumerate(train_dataloader):
    # visualization of input image
    # toPILImage = transforms.ToPILImage()
    images = images.permute(0,2,3,1)
    images = to_numpy(images)
    images = images * 0.5 + 0.5
    images = images * 255
    for id, (image, label, label_len) in enumerate(zip(images, labels, label_lens)):
      image = Image.fromarray(np.uint8(image))
      # image = toPILImage(image)
      image.show()
      print(image.size)
      print(labels2strs(label, train_dataset.id2char, train_dataset.char2id))
      print(label_len.item())
      input()


if __name__ == "__main__":
    test()