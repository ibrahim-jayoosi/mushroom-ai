import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# تحميل الموديل
model = tf.keras.models.load_model("mushroom_model.h5")

# إعداد الصفحة
st.set_page_config(
    page_title="Mushroom AI Detector",
    page_icon="🍄",
    layout="centered"
)

# CSS تصميم احترافي
st.markdown("""
<style>

.main-title{
font-size:40px;
font-weight:bold;
text-align:center;
color:#2E7D32;
}

.subtitle{
text-align:center;
color:gray;
margin-bottom:30px;
}

.card{
background-color:#f8f9fa;
padding:25px;
border-radius:15px;
box-shadow:0px 4px 15px rgba(0,0,0,0.1);
}

.result{
font-size:28px;
font-weight:bold;
text-align:center;
}

</style>
""", unsafe_allow_html=True)

# اختيار اللغة
lang = st.selectbox("🌍 Language / اللغة", ["English", "العربية"])

# النصوص
if lang == "English":

    title = "🍄 Mushroom Toxicity Detector"
    subtitle = "Artificial Intelligence for Mushroom Safety"
    upload = "Upload a mushroom image"
    result = "Toxicity Probability"
    high = "⚠️ High Toxicity"
    medium = "⚠️ Medium Toxicity"
    low = "✅ Low Toxicity"
    note = "Educational tool only"

else:

    title = "🍄 كاشف سميّة الفطر"
    subtitle = "ذكاء اصطناعي لتحليل الفطر"
    upload = "ارفع صورة فطر"
    result = "نسبة السميّة"
    high = "⚠️ سميّة عالية"
    medium = "⚠️ سميّة متوسطة"
    low = "✅ سميّة منخفضة"
    note = "للاستخدام التعليمي فقط"

# العنوان
st.markdown(f'<p class="main-title">{title}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{subtitle}</p>', unsafe_allow_html=True)

# رفع الصورة
uploaded_file = st.file_uploader(upload, type=["jpg","jpeg","png"])

if uploaded_file:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    image = Image.open(uploaded_file).resize((224,224))
    st.image(image, use_container_width=True)

    img = np.array(image)/255.0
    img = np.expand_dims(img,0)

    prediction = model.predict(img)[0][0]
    toxicity = prediction * 100

    st.markdown("---")

    st.markdown(f'<p class="result">{result}: {toxicity:.2f}%</p>', unsafe_allow_html=True)

    st.progress(int(toxicity))

    if toxicity > 70:
        st.error(high)
    elif toxicity > 40:
        st.warning(medium)
    else:
        st.success(low)

    st.caption(note)

    st.markdown('</div>', unsafe_allow_html=True)