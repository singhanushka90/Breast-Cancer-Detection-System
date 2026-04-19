import streamlit as st
import pickle
import numpy as np

model=pickle.load(open('breast_cancer_model.pkl','rb'))
scaler=pickle.load(open('breast_cancer_scaler.pkl','rb'))

st.set_page_config(page_title="Breast Cancer Survival Prediction",layout="centered")

st.markdown("""
<h1 style='text-align : center; color: #e75480;'>
             Breast Cancer Survival Prediction
            </h1>
            <p style='text-align : center; color: gray ;'>
            Fill in patient details to predict survival outcome
            </p>
            <hr>
            """,unsafe_allow_html=True)

col1,col2=st.columns(2)

with col1:
    st.subheader("Cancer Staging")
    t_stage=st.selectbox("T Stage",[0,1,2,3],
                         help="T1=0 T2=1,T3=2,T4=3")
    n_stage=st.selectbox("N Stage",[0,1,2],
                         help="N1=0,N2=1,N3=2")
    sixth_stage=st.selectbox("6th Stage",[2,3],
                             help="Cancer stage classification")
    grade=st.selectbox("Grade",[1,2,3,4])
    a_stage=st.selectbox("A stage",[0,1],
                         help="Regional=0 , Distant=1")
    
with col2:
    st.subheader("Clinical Details")
    tumor_size=st.number_input('Tumor size ',min_value=1,max_value=150,value=30)
    estrogen=st.selectbox("Estrogen Status",[1,0],
                          help="Positive =1,Negative=0")
    progesterone=st.selectbox("Progesterone Status",[1,0],
                              help="Positive=1,Negative=0")
    regional_node=st.number_input("Regional node examined",min_value=1,max_value=61,value=10)
    node_positive=st.number_input("Regional node positive",min_value=0,max_value=46,value=1)


st.subheader("Survival Timeline")
survival_months=st.slider("Survival Months",min_value=1,max_value=107,value=60)
st.markdown("<hr>",unsafe_allow_html=True)


stage_map={
    'IIA':2,'IIB':2,
    'IIIA':3,'IIIB':3,'IIIC':3
}


if st.button("Predict Survival",use_container_width=True):


    input_data=np.array([[
        t_stage,
        n_stage ,                    
        sixth_stage,                   
        grade,
        a_stage,               
        tumor_size,              
        estrogen,             
        progesterone,         
        regional_node,     
        node_positive,      
        survival_months
    ]])
    scaled=scaler.transform(input_data)
    pred=model.predict(scaled)[0]
    prob=model.predict_proba(scaled)[0]

    st.markdown("<hr>",unsafe_allow_html=True)

    if pred==1:
        st.success(f"""
                   ### Patient likely to survival !
                   **Survival Probability : {prob[1]*100:.1f}%**""")
        st.balloons()
    else:
        st.error(f"""
                 ##High Risk Patient
                 **Mortality Probability : {prob[0]*100:.1f}%**
                 Please consult specialist immediately.""")
        
st.markdown("""
            <hr>
            <p style='text-align: center; color: gray; font-size: 12px;'>
            This tool is for educational purpose only.
            Always consult a medical professional .
            </p>""",unsafe_allow_html=True)
