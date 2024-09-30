import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from PIL import Image


st.title('Hotdog or Not Hotdog')

# st.button('Button')

# st.color_picker('Color picker')

# st.checkbox('check')

# st.slider('Slider')

# st.text_input('Text input')

# st.number_input('Number input')

# st.date_input('Date input')

# st.time_input('Time input')


# st.selectbox('Select box', ['Option 1', 'Option 2', 'Option 3'])

# st.code('print("Ahmed")')

# st.balloons()

# st.snow()

# st.toast()


# chart_data = pd.DataFrame(
#     np.random.randn(20, 3), columns=["col1", "col2", "col3"]
# )
# chart_data["col4"] = np.random.choice(["A", "B", "C"], 20)

# st.scatter_chart(
#     chart_data,
#     x="col1",
#     y="col2",
#     color="col4",
#     size="col3",
# )





classifier = pipeline("image-classification",model='julien-c/hotdog-not-hotdog')

upload_file = st.file_uploader('Choose a file...')
if upload_file is not None:
    img = Image.open(upload_file)
    pred = classifier(img)
    st.write(pred) 

