import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from traitement import clust_km, segmentation

st.title("Segmentation d’image par K-means")

uploaded_file = st.file_uploader("Charge une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = plt.imread(uploaded_file)

    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img / 255.0
    st.subheader("Image originale")
    st.image(img)

    n_clusters = st.slider("Nombre de clusters", 2, 100)

    if st.button("Lancer la segmentation"):
        progress = st.progress(0)
        status = st.empty()

        status.text("Clustering des pixels...")
        km = clust_km(img, n_clusters)
        progress.progress(50)

        status.text("Segmentation de l’image...")
        img_seg = segmentation(img, km)
        progress.progress(100)

        status.text("Terminé ✅")


        st.subheader("Image segmentée")
        st.image(img_seg)

