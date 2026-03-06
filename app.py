
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Clasificador de textos")


# Cargar modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_texto.pkl")

pipeline = load_model()

# Cargar CSV de validación
@st.cache_data
def load_data():
    return pd.read_csv("datos_validacion_streamlit.csv")

df_val = load_data()

# solo columna texto
textos = df_val["texto"].astype(str)

# Selector de texto
texto_seleccionado = st.selectbox(
    "Selecciona un texto de validación o escribe uno totalmente nuevo en la celda de abajo",
    textos
)

# guardar selección
if "texto_input" not in st.session_state:
    st.session_state["texto_input"] = texto_seleccionado

# actualizar cuando cambie selección
st.session_state["texto_input"] = texto_seleccionado

# Cuadro de texto editable
texto = st.text_area(
    "Texto para clasificar",
    value=st.session_state["texto_input"],
    height=150
)

# Botón clasificar
if st.button("Clasificar"):

    if texto=="":
        st.success("El texto no puede estar vacio")
    else:
        pred = pipeline.predict([texto])[0]
        st.success(f"Predicción: {pred}")
        probs = pipeline.predict_proba([texto])[0]
        clases = pipeline.classes_
        top_idx = np.argsort(probs)[::-1][:5]
        top5 = pd.DataFrame({
                "ODS": [clases[i] for i in top_idx],
                "Probabilidad": [probs[i] for i in top_idx],
        })

        top5["Probabilidad"] = top5["Probabilidad"].map(lambda x: round(float(x), 4))
        st.subheader("Top 5 predicciones")
        st.dataframe(top5, use_container_width=True)

