from flask import Flask, jsonify, abort, make_response, request
import cv2
import numpy as np
import requests
import shutil
from urllib import parse
import json
from bs4 import BeautifulSoup
import requests
from bs4 import BeautifulSoup
import os
import pprint
import time
import urllib.error
import urllib.request
from PIL import Image
from facenet.src import facenet
import tensorflow as tf
import numpy as np
from PIL import Image
import numpy as np
import re
import glob

class FaceEmbedding(object):

    def __init__(self, model_path):
        # モデルを読み込んでグラフに展開
        facenet.load_model(model_path)

        self.input_image_size = 160
        self.sess = tf.Session()
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def __del__(self):
        self.sess.close()

    def load_image(self, image_path, width, height, mode):
        image = Image.open(image_path)
        image = image.resize([width, height], Image.BILINEAR)
        return np.array(image.convert(mode))

    def face_embeddings(self, image_path):
        image = self.load_image(image_path, self.input_image_size, self.input_image_size, 'RGB')
        prewhitened = facenet.prewhiten(image)
        prewhitened = prewhitened.reshape(-1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
        feed_dict = { self.images_placeholder: prewhitened, self.phase_train_placeholder: False }
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings

class Recognizer():
  def __init__(self):
    self.cascade_path = 'haarcascade_frontalface_default.xml'
    self.faceCascade = cv2.CascadeClassifier(self.cascade_path)
    self.path_dict = {}
    self.resizedimagefilename_dict = {}
    self.results_dict = {}

  def get_images_link_by_google_search(self, keyword):
    GOOGLE_SEARCH_URL = 'https://www.google.co.jp/search'
    params = parse.urlencode({
        'q': keyword,
        'tbm':'isch',
        'filter':'0'
    })
    google_search_url = GOOGLE_SEARCH_URL + '?' + params
    res = requests.get(google_search_url)
    soup = BeautifulSoup(res.text)
    links = soup.find_all('img')[1:]
    tmp = [links[i]['src'] for i in range(len(links))]
    links = tmp
    self.path_dict[keyword] = links
    return links

  def download_file(self, url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)
  def download_images(self, path, keyword):
    index = 0
    imagefilename_list = []
    for each_path in path:
      index += 1
      dst_path = '{}{}.jpg'.format(keyword, index)
      imagefilename_list.append(dst_path)
      self.download_file(each_path, 'images/{}'.format(dst_path))
      print('downloaded {}{}.jpg'.format(keyword, index))
    return imagefilename_list

  def cut_images(self, imagefilename_list):
    cuttedimagefilename_list = []
    for imagefilename in imagefilename_list:
        img = cv2.imread('images/{}'.format(imagefilename), cv2.IMREAD_COLOR)
        if img is None:
              print('{}:NoFace'.format(imagefilename))
        else:
          gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          face = self.faceCascade.detectMultiScale(gray, 1.1, 3)

          if len(face) > 0:
              for rect in face:
                  # 顔認識部分を赤線で囲み保存(今はこの部分は必要ない)
                  # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
                  # cv2.imwrite('detected.jpg', img)
                  x = rect[0]
                  y = rect[1]
                  w = rect[2]
                  h = rect[3]
                  cuttedimagefilename_list.append('cutted_{}'.format(imagefilename))
                  cv2.imwrite('images/cutted_{}'.format(imagefilename), img[y:y+h,  x:x+w])

          else:
              print('{}:NoFace'.format(imagefilename))

    return cuttedimagefilename_list

  def resize_pic160(self, cuttedimagefilename_list, validate_size=50):
    resizedimagefilename_list = []
    for cuttedimagefilename in cuttedimagefilename_list:
      img = Image.open('images/{}'.format(cuttedimagefilename))
      cuttedimagefilename = cuttedimagefilename.replace('.jpg', '')
      if img.size[0] > validate_size and img.size[1] > validate_size:
        img_resize = img.resize((160, 160))
        img_resize.save('images/{}_resize.jpg'.format(cuttedimagefilename))
        print('resized {}.jpg {}*{} to 160*160'.format(cuttedimagefilename, img.size[0], img.size[1]))
        resizedimagefilename_list.append('images/{}_resize.jpg'.format(cuttedimagefilename))
      else:
        print('{}.jpg {}*{} is too small'.format(cuttedimagefilename, img.size[0], img.size[1]))
    return resizedimagefilename_list

  def cos_sim(self, v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

  def get_vector_by_keyword(self, keyword):
    path = self.get_images_link_by_google_search(keyword)
    imagefilename_list = self.download_images(path, keyword)
    cuttedimagefilename_list = self.cut_images(imagefilename_list)
    resizedimagefilename_list = self.resize_pic160(cuttedimagefilename_list)
    files = resizedimagefilename_list
    self.resizedimagefilename_dict[keyword] = resizedimagefilename_list

    FACE_MEDEL_PATH = 'facenet/src/models/20180402-114759/20180402-114759.pb'
    face_embedding = FaceEmbedding(FACE_MEDEL_PATH)

    embedded_dict = {}
    embedded_list = []
    for file in files:
      tmp = face_embedding.face_embeddings(file)[0]
      embedded_dict['{}'.format(file)] = tmp
      embedded_list.append(tmp)

    return embedded_list, embedded_dict

  def get_vector_by_url(self, compare_link, keyword):
    compare_path = []
    compare_path.append(compare_link)
    compare_imagefilename_list = self.download_images(compare_path, keyword)
    compare_cuttedimagefilename_list = self.cut_images(compare_imagefilename_list)
    compare_resizedimagefilename_list = self.resize_pic160(compare_cuttedimagefilename_list, 10)

    FACE_MEDEL_PATH = 'facenet/src/models/20180402-114759/20180402-114759.pb'
    face_embedding = FaceEmbedding(FACE_MEDEL_PATH)
    file = compare_resizedimagefilename_list[0]

    return face_embedding.face_embeddings(file)[0]

  def compare(self, compare_keyword, compare_url, keywords):
    compare = self.get_vector_by_url(compare_url, compare_keyword)
    for keyword in keywords:
      print(keyword)
      embedded_list, embedded_dict = self.get_vector_by_keyword(keyword)
      resizedimagefilename_list = self.resizedimagefilename_dict[keyword]
      resizedimagefilename_list = [re.sub(r"\D", "", s) for s in resizedimagefilename_list]
      self.results_dict[keyword] = {}
      index = 0
      for embedded_vector in embedded_list:
        print(index+1)
        print(self.cos_sim(embedded_vector, compare))
        print(self.path_dict[keyword][int(resizedimagefilename_list[index])-1])
        self.results_dict[keyword][self.path_dict[keyword][int(resizedimagefilename_list[index])-1]] = self.cos_sim(embedded_vector, compare)
        index += 1
    return self.results_dict

  def register(self, url, keyword, save=True):
    os.makedirs('images/', exist_ok=True)
    vec = self.get_vector_by_url(url, keyword)
    shutil.rmtree('images/')
    if save:
      os.makedirs('vector_data/{}'.format(keyword), exist_ok=True)
      url = url.replace(':', 'krkrkr').replace('/', 'srsrsr').replace('.', 'dtdtdt').replace('?', 'queque').replace('=', 'eqeq')
      np.save('vector_data/{}/{}'.format(keyword, url), vec)
    return vec

  def compare_with_registerd_data(self, url, keyword):
    vec = self.register(url, keyword, False)
    keyword_dirs = glob.glob('vector_data/*')
    cos_dic = {}
    cos_dic[keyword] = url
    cos_dic['compare'] = {}
    for keyword_dir in keyword_dirs:
        cos_dic['compare'][keyword_dir] = {}
        np_files = glob.glob('{}/*'.format(keyword_dir))
        for np_file in np_files:
          registerd_vec = np.load(np_file)
          cos = self.cos_sim(registerd_vec, vec)
          url_link = np_file.replace('krkrkr', ':').replace('srsrsr', '/').replace('dtdtdt', '.').replace('queque', '?').replace('eqeq', '=').replace('.npy', '')
          cos_dic['compare'][keyword_dir][url_link] = str(cos)
    # np.save('vector_data/{}'.format(keyword), vec)
    return cos_dic


api = Flask(__name__)

@api.route('/register', methods=['POST'])
def register():
  data = request.get_json()
  print(data)
  url = data['url']
  keyword = data['keyword']
  recog = Recognizer()
  vec = recog.register(url, keyword)
  response = {"status":"登録完了"}
  return make_response(jsonify(response))

@api.route('/compare', methods=['POST'])
def compare():
  data = request.get_json()
  print(data)
  url = data['url']
  keyword = data['keyword']
  recog = Recognizer()
  cos_dic = recog.compare_with_registerd_data(url, keyword)
  return make_response(jsonify(cos_dic))


@api.route('/result/<string:Id>', methods=['GET'])
def get_result(Id):
    # os.makedirs('images', exist_ok=True)
    # compare_keyword = 'kawatahina'
    # compare_url = 'https://thetv.jp/i/nw/1023326/10207156.jpg?w=615'
    # keywords = ['saitokyoko', 'kosakanao']
    # recog = Recognizer()
    # recog.compare(compare_keyword, compare_url, keywords)
    # shutil.rmtree('images/')

    # d = recog.results_dict
    # tmp = {}
    # for key in d:
    #     tmp[key] = {link:str(d[key][link]) for link in d[key]}

    print(Id)
    tmp = {}
    tmp['a'] = {}
    tmp['a']['al'] = '7'
    res_data = json.dumps(tmp)
    # return make_response(jsonify(tmp))
    return jsonify(tmp)

@api.route('/results', methods=['POST'])
def get_result_by_post_rq():
    data = request.get_json()
    print(data)
    os.makedirs('images', exist_ok=True)
    compare_keyword = data['compare_keyword']
    compare_url = data['compare_url']
    keywords = []
    for key in data['keywords'].keys():
        keywords.append(data['keywords'][key])
    recog = Recognizer()
    recog.compare(compare_keyword, compare_url, keywords)
    shutil.rmtree('images/')

    d = recog.results_dict
    tmp = {}
    for key in d:
        tmp[key] = {link:str(d[key][link]) for link in d[key]}

    return make_response(jsonify(tmp))
    # requests.post(url, json=data)を受け取る
    # print(type(data))

if __name__ == '__main__':
    api.run(debug=True)
