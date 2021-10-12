import streamlit as st
import torch
from PIL import Image
from utils import predict, show_tensor_images

st.set_page_config(page_title="GAN App",page_icon=":fire")


st.sidebar.title('About the App')
st.sidebar.write("This is an app that transforms a horse to a zebra.")
st.sidebar.write("The app is a GAN model that converts picture of a horse to that of a zebra.")
st.sidebar.write("Creepy, right...?")
st.sidebar.write("The model is not perfect so there may be some mistakes.")
st.sidebar.write("And don't upload a picture that is not a horse, unless you want to see some weird stuff...")
st.sidebar.write("No horse image at the moment, no worries, press the 'Test' button to test the model with a sample picture.")




#start the user interface
st.title("Horse-Zebra App")
st.write("Upload an image of a horse, and pres 'Transform' to change it.")
st.write("PS: If on mobile, switch to Desktop mode for better display.")

st.write("No horse image at the moment, no worries, press the 'Test' button below to test the model with a sample picture")
if st.button("Test", key='test'):
        real_image, output = predict(Image.open("demo.jpg").resize((256,256)))
        output = show_tensor_images(torch.cat([real_image, output]), size=(3, 256, 256))

        st.image(output)
        st.write("Before and After")

upload_image = st.file_uploader(label='Select your horse image...', type=('png', 'jpg', 'jpeg'), key='cimage')


if upload_image is not None:
    
    if st.button("Transform", key='transform'):
        real_image, output = predict(Image.open(upload_image).resize((256,256)))
        output = show_tensor_images(torch.cat([real_image, output]), size=(3, 256, 256))

        st.image(output)
        st.write("Before and After")








#custom footer courtesy of Heflin_Stephen_Raj_S, https://discuss.streamlit.io/t/streamlit-footer/12181
footer="""
<style>

    .footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
    }
    .footer p{
        margin: 0;
    }
</style>
<div class="footer">
    <p>Developed with Streamlit by Ibrahim Adebayo.</p>
    <p>Adebayoibrahim2468@gmail.com</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)