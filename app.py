import streamlit as st
import torch
from PIL import Image
from utils import predict, show_tensor_images

st.set_page_config(page_title="GAN App",page_icon=":fire")


st.sidebar.title('About the App')
st.sidebar.write("This is an app that transforms a horse to a zebra, or a zebra to a horse.")
st.sidebar.write("The app consists of two GAN models, one converts picture of a horse to that of a zebra and the other converts picture of a zebra to that of a horse.")
st.sidebar.write("Creepy, right...?")
st.sidebar.write("The model is not perfect so there may be some mistakes (The model converting horse to zebra is better than the one converting zebra to horse).")
st.sidebar.write("And don't upload a picture that is not a horse or that is not a zebra, unless you want to see some weird stuff...")




#start the user interface
st.title("Horse-Zebra and Zebra-Horse App")
st.write("Upload an image of either a horse or a zebra.")
st.write("Choose which model to use and press 'Transform' to change it.")
st.write("PS: If on mobile, switch to Desktop mode for better display.")

upload_image = st.file_uploader(label='Select your horse/zebra image...', type=('png', 'jpg', 'jpeg'), key='cimage')
model_type = st.selectbox("Select model to use...", ["Horse to Zebra", "Zebra to Horse"], key="model")


if upload_image is not None:
    
    if st.button("Transform", key='transform'):
        real_image, output = predict(Image.open(upload_image).resize((256,256)), model_type=model_type)
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