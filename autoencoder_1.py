import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
#Imports
f = open('train.pickle', 'rb')
train_data,labels=pickle.load(f)

print(train_data)
#Sampling Function from gaussian
def sample(mean,logvar,shape=(1,500)):
	z_mean,z_log_variance=mean,logvar
	print(z_mean.shape)
	epsillon=np.random.normal(size=shape,loc=0.0,scale=1.0)
	return z_mean+tf.exp(z_log_variance/2)*epsillon

#Create weights using get_variable
def create_weights(shape,name):
    weights=tf.get_variable(name,shape,initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
    return weights

#Create biases using get_variable
def create_biases(shape,name):
    bias=tf.get_variable(name,shape=shape,initializer=tf.zeros_initializer(),dtype=tf.float64)
    return bias

def create_FClayer(X,weight,bias):
    return tf.matmul(X,weight)+bias


#hyperparameters
latent_dim=20
hidden_dim=500
input_size=800
mse_weight=500
batch_size=30
iterations=100000
recording_interval=1000
saving_interval=10000
###############

X=tf.placeholder(tf.float64,shape=[None,input_size])


w_1=create_weights(shape=[input_size,hidden_dim],name='w_1')
b_1=create_biases(shape=[hidden_dim],name='b_1')
hidden_layer=tf.nn.tanh(create_FClayer(X,w_1,b_1))

#Encoder
w_mean=create_weights(shape=[hidden_dim,latent_dim],name='w_mean')
b_mean=create_weights(shape=[latent_dim],name='b_mean')
mean=create_FClayer(hidden_layer,w_mean,b_mean)

w_logstd=create_weights(shape=[hidden_dim,latent_dim],name='w_logstd')
b_logstd=create_weights(shape=[latent_dim],name='b_logstd')
logstd=create_FClayer(hidden_layer,w_logstd,b_logstd)

z=sample(mean,logstd,shape=[1,latent_dim])


# noise=tf.random_normal(shape=[1,latent_dim])

# z=mean+tf.multiply(tf.exp(0.5*logstd),noise)


#Decoder
w_2=create_weights(shape=[latent_dim,hidden_dim],name='w_2')
b_2=create_biases(shape=[hidden_dim],name='b_2')
hidden_layer_2=tf.nn.tanh(create_FClayer(z,w_2,b_2))

w_3=create_weights(shape=[hidden_dim,input_size],name='w_3')
b_3=create_weights(shape=[input_size],name='b_3')
reconstruction=create_FClayer(hidden_layer_2,w_3,b_3)

mse=tf.reduce_mean(tf.square(X-reconstruction))
KL_term=-0.5*tf.reduce_mean(1+2*logstd-tf.square(mean)-tf.exp(2*logstd),reduction_indices=1)

variational_lower_bound=tf.reduce_mean(mse_weight*mse+KL_term)
optimizer=tf.train.AdamOptimizer(0.000001).minimize(variational_lower_bound)

init=tf.global_variables_initializer()
sess=tf.InteractiveSession()
sess.run(init)
saver=tf.train.Saver()

variational_lower_bound_array=[]
mse_array=[]
KL_term_array=[]
iteration_array = [i*recording_interval for i in range(iterations//recording_interval)]
for i in range(iterations):
    x_batch=train_data[(i*batch_size)%3500:((i+1)*batch_size)%3500]
    loss1,_=sess.run([variational_lower_bound,optimizer],feed_dict={X:x_batch})

    if (i%recording_interval == 0):
        #every 1K iterations record these values
        vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
        print ("Iteration: {}, Loss: {}".format(i, vlb_eval))
        variational_lower_bound_array.append(vlb_eval)
        mse_array.append(np.mean(mse.eval(feed_dict={X: x_batch})))
        KL_term_array.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))
    if i%saving_interval==0:
    	save_path = saver.save(sess, "checkpoints/model_{}.ckpt".format(i))
    	print("Model saved in path: on iteration %s" % save_path)