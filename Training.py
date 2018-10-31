import tensorflow as tf

from _utils import *
from Variables import CONFIG
import numpy as np

def content_cost(a_C, a_G):                                                             //computes content cost
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.transpose(tf.reshape(a_C,shape=(m,n_H*n_W,n_C)))
    a_G_unrolled = tf.transpose(tf.reshape(a_G,shape=(m,n_H*n_W,n_C)))
    J_content = tf.reduce_sum(tf.square(a_C_unrolled-a_G_unrolled))/(4*n_H*n_W*n_C)
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.transpose(tf.reshape(a_S,shape=(n_H*n_W,n_C)))
    a_G = tf.transpose(tf.reshape(a_G,shape=(n_H*n_W,n_C)))
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG)))/(4*(n_C*n_H*n_W)**2)
    return J_style_layer

def compute_style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    return alpha*J_content+beta*J_style

content_image = load_image(CONFIG.CONTENT_IMAGE)
style_image = load_image(CONFIG.STYLE_IMAGE)
genrated_image = genrate_image()

sess = tf.Session()

model = load_vgg_model(CONFIG.VGG_MODEL)

sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = content_cost(a_C, a_G)

sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, CONFIG.STYLE_LAYERS)
J = total_cost(J_content, J_style,alpha=CONFIG.ALPHA,beta=CONFIG.BETA)

optimizer = tf.train.AdamOptimizer(CONFIG.LEARNINF_RATE)
train_step = optimizer.minimize(J)

sess.run(tf.global_variables_initializer())
sess.run(model['input'].assign(genrated_image))
for i in range(CONFIG.NUM_ITERATIONS):
	_ = sess.run(train_step)
	generated_image = sess.run(model['input'])
	if i%20 == 0:
	    Jt, Jc, Js = sess.run([J, J_content, J_style])
	    print("Iteration " + str(i) + " :")
	    print("total cost = " + str(Jt))
	    print("content cost = " + str(Jc))
	    print("style cost = " + str(Js))
	    save_image(CONFIG.OUTPUT_DIR +'/'+ str(i) + ".png", generated_image)
save_image(CONFIG.OUTPUT_DIR+'/generated_image.jpg', generated_image)
