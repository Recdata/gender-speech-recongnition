#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre_process
import sklearn.metrics as sme
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm as sv
import sklearn.model_selection as ms
from glob import glob
import sklearn.preprocessing as sp
encoding=sp.OneHotEncoder()


# In[2]:


audio = 'new'
audio_b='girl'
files= glob(audio+'/*.wav')
files1= glob(audio_b+'/*.wav')

i = 0
df= pd.DataFrame(np.arange(13).reshape(1,-1))

a=[]
for i in files1:
    b = i[5:]
    c=f"new\\{b}"
    a.append(c)

for i in files:
    y1, sr1 = librosa.load(i)
    
    ab = pd.DataFrame(librosa.feature.mfcc(y1, sr1,n_mfcc=1))
    #ab = ab.T
    ab= ab.iloc[:,0:13]
    if i in a:
        ab['label'] = 'girl'
    else:
        ab['label'] = 'boy'

    df = pd.concat([df,ab],axis=0)

df.dropna(inplace=True)


# In[3]:


df.head(10)


# In[4]:




li_test = 'aman.wav'
dft =pd.DataFrame([])
y2, sr2 = librosa.load(li_test)
y2 = y2[2000:2500]
ab1 = pd.DataFrame(librosa.feature.mfcc(y = y2, sr = sr2))
ab1 = ab1.T
ab1= ab1.iloc[:,0:13]
ab1['label']='boy'
dft = pd.concat([dft,ab1],axis=0)
dft


# In[5]:


x = df.iloc[:,0:-1].values
x = np.c_[x]

y = np.c_[LabelEncoder().fit_transform(df.iloc[:,-1].values)]


# In[6]:


x1 = dft.iloc[:,0:-1].values

y1 = np.c_[LabelEncoder().fit_transform(dft.iloc[:,-1].values)]


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)
from sklearn .utils import all_estimators
estimators = all_estimators(type_filter='classifier')

li = []
nam = []
ac = []
pr = []
for name,get_model in estimators:
    try:
        
        model = get_model()
        model.fit(x_train,y_train)
        predicted_y = model.predict(x_test)
        display(name)
        nam.append(name)
        print(sme.classification_report(y_test,predicted_y))
        
        f1 = sme.f1_score(y_test,predicted_y, sample_weight=None, pos_label='positive',average='micro')
        pre = sme.precision_score(y_test,predicted_y, average='micro')
        acc = sme.accuracy_score(y_test,predicted_y, normalize=False)

        print('f-1 score',f1)
        print('precision score',pre)
        print('accuracy score',acc)
    
        li.append(f1)
        ac.append(acc)
        pr.append(pre)
    except Exception as e:
        print('Unable to import: ', name)
        print(e)
li
nam
dic={
    "name":nam,
    'f-1 score':li,
    'precision score':pr,
    'accuracy score':ac,
}
dftr= pd.DataFrame.from_dict(dic, orient='index')


# In[8]:


dftr = dftr.T
dftr


# In[9]:


dftr[['f-1 score','name']].sort_values(by='f-1 score', ascending=False)


# In[10]:


model = lm.SGDClassifier()
model.fit(x_train, y_train)


# In[11]:


predicted_y1 = model.predict(x1)
predicted_y1
# predicted_y1 = LabelEncoder.transform(predicted_y1)
# L = le.inverse_transform(predicted_y1)
# predicted_y1


# In[18]:



data = pd.DataFrame(np.array(1).reshape(-1,1), columns=['label'])

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

df['label'] = LabelEncoder().fit_transform(df['label'])
le.fit_transform(data['label'])


# In[19]:


model = sv.SVC(kernel='poly')
model.fit(x_train, y_train)

model = sv.SVC(kernel='linear')
model.fit(x_train, y_train)

#PREDICT MODEL
X_new = x1
predicted_y1 = model.predict(X_new)
predicted_y1

#encoding.inverse_transform(predicted_y1)
#print(sme.classification_report(y1,predicted_y1))


# In[27]:


l = [0]
print(sme.classification_report(y1,predicted_y1))
LabelEncoder.transform(l[0])

# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

# Sampling frequency
freq = 44100

# Recording duration
duration = 5

# Start recorder with the given values
# of duration and sample frequency
recording = sd.rec(int(duration * freq),samplerate=freq, channels=2)

# Record audio for the given number of seconds
sd.wait()

# This will convert the NumPy array to an audio
# file with the given sampling frequency
write("recording0.wav", freq, recording)

# Convert the NumPy array to audio file
wv.write("aman.wav", recording, freq, sampwidth=2)
import numpy as np
import simpleaudio as sa

frequency = 440  # Our played note will be 440 Hz
fs = 44100  # 44100 samples per second
seconds = 3  # Note duration of 3 seconds

# Generate array with seconds*sample_rate steps, ranging between 0 and seconds
t = np.linspace(0, seconds, seconds * fs, False)

# Generate a 440 Hz sine wave
note = np.sin(frequency * t * 2 * np.pi)

# Ensure that highest value is in 16-bit range
audio = note * (2**15 - 1) / np.max(np.abs(note))
# Convert to 16-bit data
audio = audio.astype(np.int16)

# Start playback
play_obj = sa.play_buffer(audio, 1, 2, fs)

# Wait for playback to finish before exiting
play_obj.wait_done()
# In[ ]:





# In[15]:


audio = 'new'
audio_b='girl'
files= glob(audio+'/*.wav')
files1= glob(audio_b+'/*.wav')

i = 0
df= pd.DataFrame(np.arange(13).reshape(1,-1))

a=[]
for i in files1:
    b = i[5:]
    c=f"new\\{b}"
    a.append(c)

for i in files:
    y1, sr1 = librosa.load(i)
    
    ab = pd.DataFrame(librosa.feature.mfcc(y1, sr1,n_mfcc=1))
    #ab = ab.T
    ab= ab.iloc[:,0:13]
    if i in a:
        ab['label'] = 'girl'
    else:
        ab['label'] = 'boy'

    df = pd.concat([df,ab],axis=0)

df.dropna(inplace=True)

#
li_test = 'aman.wav'
dft =pd.DataFrame([])
y2, sr2 = librosa.load(li_test)
y2 = y2[2000:2500]
ab1 = pd.DataFrame(librosa.feature.mfcc(y = y2, sr = sr2))
ab1 = ab1.T
ab1= ab1.iloc[:,0:13]
ab1['label']='boy'
dft = pd.concat([dft,ab1],axis=0)


#
x = df.iloc[:,0:-1].values
x = np.c_[x]

y = np.c_[LabelEncoder().fit_transform(df.iloc[:,-1].values)]

#
x1 = dft.iloc[:,0:-1].values

y1 = np.c_[LabelEncoder().fit_transform(dft.iloc[:,-1].values)]

#
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)

#
model = sv.SVC(kernel='poly')
model.fit(x_train, y_train)

model = sv.SVC(kernel='linear')
model.fit(x_train, y_train)

#PREDICT MODEL
X_new = x1
predicted_y1 = model.predict(X_new)
predicted_y1


# In[ ]:


predicted_y1.

