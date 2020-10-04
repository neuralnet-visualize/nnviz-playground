#!/usr/bin/python3

import streamlit as st
from neuralnet_visualize import visualize as nnviz

st.title("Interactive Playground")

net = nnviz.visualizer()

net.add_layer('dense', 5)
net.add_layer('dense', 13)
net.add_layer('dense', 8)

st.graphviz_chart(net.visualize(give_obj=True))