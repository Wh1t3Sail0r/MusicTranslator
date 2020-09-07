from flask import Flask,request,send_from_directory,render_template
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
app = Flask(__name__, static_url_path='')
tf.disable_v2_behavior()

# Get scale based on key
def get_scale(key):
    scales = open("scales.txt", "r")
    scale_list = []
    for scale in scales:
        if scale[0] == key:
            scale = scale.strip()
            scale_list = scale.split(" ")

    return scale_list

# Map western to carnatic based on key signature
def translate(key, notes):
    # Get scale
    scale = get_scale(key)
    carn = []

    for note in notes:
        if note == scale[0]:
            carn.append('S')
        elif note == scale[1]:
            carn.append('R1')
        elif note == scale[2]:
            carn.append('R1')
        elif note == scale[3]:
            if carn[len(carn) - 1] != 'R1':
                carn.append('R2')
            else:
                carn.append('G1')
        elif note == scale[4]:
            if carn[len(carn) - 1] != 'R2' and carn[len(carn) - 1] != 'R1':
                carn.append('R3')
            else:
                carn.append('G2')
        elif note == scale[5]:
            if carn[len(carn) - 1] != 'R2' and carn[len(carn) - 1] != 'R1':
                carn.append('R3')
            else:
                carn.append('G2')
        elif note == scale[6]:
            carn.append('G3')
        elif note == scale[7]:
            carn.append('G3')
        elif note == scale[8]:
            carn.append('M1')
        elif note == scale[9]:
            carn.append('M1')
        elif note == scale[10]:
            carn.append('M2')
        elif note == scale[11]:
            carn.append('M2')
        elif note == scale[12]:
            carn.append('P')
        elif note == scale[13]:
            carn.append('D1')
        elif note == scale[14]:
            carn.append('D1')
        elif note == scale[15]:
            if carn[len(carn) - 1] != 'D1':
                carn.append('D2')
            else:
                carn.append('N1')
        elif note == scale[16]:
            if carn[len(carn) - 1] != 'D2' and carn[len(carn) - 1] != 'D1':
                carn.append('D3')
            else:
                carn.append('N2')
        elif note == scale[17]:
            if carn[len(carn) - 1] != 'D2' and carn[len(carn) - 1] != 'D1':
                carn.append('D3')
            else:
                carn.append('N2')
        elif note == scale[18]:
            carn.append('N3')
        elif note == scale[19]:
            carn.append('N3')
        elif note == scale[20]:
            carn.append('S')
        elif note == scale[21]:
            carn.append('S')

    return carn

def sparse_tensor_to_strs(sparse_tensor):
  indices= sparse_tensor[0][0]
  values = sparse_tensor[0][1]
  dense_shape = sparse_tensor[0][2]

  strs = [ [] for i in range(dense_shape[0]) ]

  string = []
  ptr = 0
  b = 0

  for idx in range(len(indices)):
      if indices[idx][0] != b:
          strs[b] = string
          string = []
          b = indices[idx][0]

      string.append(values[ptr])

      ptr = ptr + 1

  strs[b] = string

  return strs


def normalize(image):
  return (255. - image)/255.


def resize(image, height):
  width = int(float(height * image.shape[1]) / image.shape[0])
  sample_img = cv2.resize(image, (width, height))
  return sample_img


voc_file = "vocabulary_semantic.txt"
model = "Semantic-Model/semantic_model.meta"


tf.reset_default_graph()
sess = tf.InteractiveSession()
# Read the dictionary
dict_file = open(voc_file,'r')
dict_list = dict_file.read().splitlines()
int2word = dict()
for word in dict_list:
  word_idx = len(int2word)
  int2word[word_idx] = word
dict_file.close()

# Restore weights
saver = tf.train.import_meta_graph(model)
saver.restore(sess,model[:-5])

graph = tf.get_default_graph()

input = graph.get_tensor_by_name("model_input:0")
seq_len = graph.get_tensor_by_name("seq_lengths:0")
rnn_keep_prob = graph.get_tensor_by_name("keep_prob:0")
height_tensor = graph.get_tensor_by_name("input_height:0")
width_reduction_tensor = graph.get_tensor_by_name("width_reduction:0")
logits = tf.get_collection("logits")[0]

# Constants that are saved inside the model itself
WIDTH_REDUCTION, HEIGHT = sess.run([width_reduction_tensor, height_tensor])

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)


@app.route('/img/<filename>')
def send_img(filename):
   return send_from_directory('', filename)


@app.route("/")
def root():
    return render_template('index.html')


@app.route('/translate', methods = ['GET', 'POST'])
def predict():
   if request.method == 'POST':
      f = request.files['file']
      img = f
      image = Image.open(img).convert('L')
      image = np.array(image)
      image = resize(image, HEIGHT)
      image = normalize(image)
      image = np.asarray(image).reshape(1,image.shape[0],image.shape[1],1)

      seq_lengths = [ image.shape[2] / WIDTH_REDUCTION ]
      prediction = sess.run(decoded,
                            feed_dict={
                                input: image,
                                seq_len: seq_lengths,
                                rnn_keep_prob: 1.0,
                            })
      str_predictions = sparse_tensor_to_strs(prediction)

      array_of_notes = []

      for w in str_predictions[0]:
          array_of_notes.append(int2word[w])


      notes=[]
      key_sig = array_of_notes[1][13:14]
      print(key_sig)

      for i in array_of_notes:
          if i[0:5]=="note-":
              if not i[6].isdigit():
                  notes.append(i[5:7])
              else:
                  notes.append(i[5])

      print(notes)
      carn = translate(key_sig, notes)
      print(carn)

      img = Image.open(img).convert('L')
      size = (img.size[0], int(img.size[1]*1.5))
      layer = Image.new('RGB', size, (255,255,255))
      layer.paste(img, box=None)
      img_arr = np.array(layer)
      height = int(img_arr.shape[0])
      width = int(img_arr.shape[1])
      # print(img_arr.shape[0])
      draw = ImageDraw.Draw(layer)
      # font = ImageFont.truetype(<font-file>, <font-size>)
      font = ImageFont.truetype("Aaargh.ttf", 20)
      # draw.text((x, y),"Sample Text",(r,g,b))
      j = width / 9
      for i in notes:
          draw.text((j, height-80), i, (0,0,0), font=font)
          j+= (width / (len(notes) + 4))
      j = width / 9
      for c in carn:
          draw.text((j, height-40), c, (0,0,0), font=font)
          j+= (width / (len(notes) + 4))

      layer.save("annotated.png")
      os.system('open -a "Google Chrome" /Users/shreyasravi/Desktop/Python/MusicTranslator/annotated.png')
      return render_template('index.html')


if __name__=="__main__":
    app.run()
    # scale = get_scale('C')
    # notes = ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F#', 'G']
    # print(notes)
    # print(translate('G', notes))
