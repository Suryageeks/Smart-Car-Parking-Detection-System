import streamlit as st
import torch
import torchvision
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
import pandas as pd
import requests
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from streamlit_folium import folium_static
import folium

setup_logger()

title = '<p style="font-family:monospace; color:orange; font-size: 40px;"><b>Parking Space Detection using Detectron2</b></p>'
st.markdown(title,unsafe_allow_html=True)

st.sidebar.subheader('**Upload a File**')
file_upload = st.sidebar.file_uploader("Choose a Image",type=['png','jpeg','jpg'])


if file_upload is not None:
  #get file details
  file_details = {"filename":file_upload.name, "filetype":file_upload.type,"filesize":file_upload.size}
  st.sidebar.markdown('**File Details**')
  st.sidebar.write(file_details)

  st.subheader('**Input Image**')  
  file_bytes = np.asarray(bytearray(file_upload.read()), dtype=np.uint8)  
  opencv_image = cv2.imdecode(file_bytes, 1)
  st.image(opencv_image, channels="BGR",width=380)

st.write('\n')

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file('config.yml')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  
cfg.MODEL.WEIGHTS = 'model_final.pth'
predictor = DefaultPredictor(cfg)
outputs = predictor(opencv_image)

c,k=0,0
for i in outputs["instances"].pred_classes:
  if i==1:
    c=c+1
  elif i==2:
    k=k+1

st.subheader('**Inferenced Image**')
v = Visualizer(opencv_image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
st.image(out.get_image()[:, :, ::-1],caption='Processed Image',width=380)
st.write('**Inferenced Details**')
st.markdown(f"Available Parking Space: {c}")
st.markdown(f"Filled Parking Space: {k}")

# Visualize
st.subheader('**Visualize**')
d = {'X_axis': ['Available Parking Space','Filled Parking Space'],
     'Y_axis': [c,k]}

df = pd.DataFrame(d)
fig = px.bar(        
        df,
        x = "X_axis",
        y = "Y_axis",
        title = "Stats",
        color="X_axis"        
    )
st.plotly_chart(fig)

st.markdown('**Parking Location Map**')
m = folium.Map(location=[22.5629,88.3962],width=550, height=352,zoom_start=80,zoom_control=False)
folium.Marker(
    location=[22.5629,88.3967],
    popup="Parking Available {}/{}".format(c,(c+k)),
    icon=folium.Icon(prefix="fa",color="orange",icon="car"),
    tiles='CartoDB Positron'
).add_to(m)

folium.TileLayer('Stamen Terrain').add_to(m)
folium.TileLayer('cartodbpositron').add_to(m)
folium.TileLayer('cartodbdark_matter').add_to(m)
folium.LayerControl().add_to(m)

folium_static(m)

#Message API
#API KEY
headers={
    "authorization":"xxxxxx", #put your own api key   
    "Content-Type":"application/x-www-form-urlencoded"
}

st.subheader('**Parking Kiosk Information**')
numbers= st.text_input('Enter Mobile Number')
url='https://www.fast2sms.com/dev/bulkV2'
message= f'Parking Space Available - {c}'
payload= f'sender_id=TXTIND&message={message}&route=v3&language=english&numbers={numbers}'
if st.button('Send'):
  response=requests.request('POST',url=url,data=payload,headers=headers)
  st.success('**Message Sent Successfully**')
