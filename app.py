#!/usr/bin/python3

import streamlit as st
from neuralnet_visualize import visualize as nnviz

LAYER_PARAMS = {'layer_type': ['dense', 'conv2d', 'maxpool2d', 'avgpool2d', 'flatten'], 'nodes': [1], 'filters': [1], 'kernel_size': [1, 15], 'padding': ['same', 'valid'], 'stride': [1, 15], 'pool_size': [1, 15]}

# Setting page configurations
st.beta_set_page_config(page_title="Neural Network Visualizer Playground")

st.title("Interactive Playground")

number = st.number_input('Number of Layers', min_value=2, max_value=20)

@st.cache(allow_output_mutation=True)
def give_empty_params_list(number):
    list_of_dicts = [dict() for _ in range(number)]

    return list_of_dicts

params_list = give_empty_params_list(number)

layer_num = st.selectbox("Layer Number", options=['Layer '+str(n) for n in range(1, number+1)])

st.sidebar.title("{} Parameters".format(layer_num))
for name, value in LAYER_PARAMS.items():
    if isinstance(value[0], int) and len(value) == 1:
        params_list[int(layer_num[-1])-1][name] = st.sidebar.number_input(label=name, min_value=value[0])
    elif isinstance(value[0], int) and len(value) == 2:
        params_list[int(layer_num[-1])-1][name] = st.sidebar.number_input(label=name, min_value=value[0], max_value=value[1])
    else:
        params_list[int(layer_num[-1])-1][name] = st.sidebar.selectbox(label=name, options=value)

net = nnviz.visualizer()

for n in range(number):
    if len(params_list[n]) != 0:
        net.add_layer(**params_list[n])

if st.button("Generate Neural Network"):
    try:
        st.graphviz_chart(net.visualize(give_obj=True))
    except:
        st.warning("Set the parameters of atleast 2 layers!")