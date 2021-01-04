import streamlit as st
from PIL import Image
import style
import os
import imghdr
from io import BytesIO
import base64

# style image paths:
root_style = "./images/style-images"

# general download function


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

# download image function


def get_image_download_link(img, file_name, style_name):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a style = "color:black" href="data:file/jpg;base64,{img_str}" download="{style_name+"_"+file_name+".jpg"}">Download Result</a>'
    return href

# Whenever we will use a new input image the whole streamlit application runs
# from the top and that means we load the model again even in the case where we
# are using the same model, to overcome this problem we will use caching


# st.title("Neural Style Transfer")  ## Not Recommended as HTML may soon
# be removed from streamlit
st.markdown("<h1 style='text-align: center; color: Blue;'>Neural Style Transfer</h1>",
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: right; color: Blue;'>by Divy Mohan Rai</h3>",
            unsafe_allow_html=True)

# EXPERIMENTS###################################################################
main_bg = "./images/pyto.png"
main_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
###################################################################################

# creating a side bar for picking the style of image
style_name = st.sidebar.selectbox(
    'Select Style',
    ("candy", "mosaic", "rain_princess",
     "udnie", "tg", "demon_slayer", "ben_giles", "ben_giles_2")
)
path_style = os.path.join(root_style, style_name+".jpg")


# Upload image functionality
img = None
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

show_file = st.empty()

# checking if user has uploaded any file
if not uploaded_file:
    show_file.info("Please Upload an Image")
else:
    img = Image.open(uploaded_file)
    # check required here if file is an image file
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.image(path_style, caption='Style Image', use_column_width=True)

# identifying if the file is an image
# in this case it is not explicilty needed as when we specified the type for upload file to be jpg
# jpeg and png therefore the streamlit application will only allow user to allow to upload such file
# types but still it is a good practice to have a check

extensions = [".png", ".jpeg", ".jpg"]

if uploaded_file is not None and any(extension in uploaded_file.name for extension in extensions):

    name_file = uploaded_file.name.split(".")
    root_model = "./saved_models"
    model_path = os.path.join(root_model, style_name+".pth")

    # root_input = "./images/content-images"
    # input_image = os.path.join(root_input, img)

    # png by default has 4 channels and not converting it into 3 channels will throw an error as
    # our model was trained to handle 3 channels
    img = img.convert('RGB')
    input_image = img

    root_output = "./images/output-images"
    output_image = os.path.join(
        root_output, style_name+"-"+name_file[0]+".jpg")

    stylize_button = st.button("Stylize")

    if stylize_button:
        model = style.load_model(model_path)
        stylized = style.stylize(model, input_image, output_image)
        # displaying the output image
        st.write("### Output Image")
        # image = Image.open(output_image)
        st.image(stylized, width=400, use_column_width=True)
        st.markdown(":star2:" + " " + get_image_download_link(
            stylized, name_file[0], style_name)+" "+":star2:", unsafe_allow_html=True)
