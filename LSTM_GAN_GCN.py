import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# LSTM+GAN+GCN Model for temporal link predicton of weighted dynamic networks

def v_initialization(m, n):
    #Function to initialze the weight matrix
    range = np.sqrt(6.0 / (m+n))
    initial = tf.random_uniform([m, n], minval=-range, maxval=range, dtype=tf.float32)
    return tf.Variable(initial)

def gcn_factor(adj):
    #Function to calculate the GCN factor of a certain network snapshot
    adj_ = adj + np.eye(node, node)
    r_sum = np.array(adj_.sum(1))
    inverse_sqrt = np.power(r_sum, -0.5).flatten()
    inverse_sqrt[np.isinf(inverse_sqrt)] = 0.
    matrix_inverse_sqrt = np.mat(np.diag(inverse_sqrt))
    gcn_fact = matrix_inverse_sqrt*adj_*matrix_inverse_sqrt # The GCN factor GCN因子

    return gcn_fact
def get_data(name, time_index, node, max):

    print('Read network snapshot #%d'%(time_index))
    c_adj = np.mat(np.zeros((node, node)))
    f = open('%s_%d.txt'%(name, time_index))
    line = f.readline()
    while line:
        sequence = line.split()
        #print(sequence)
        source = int(sequence[0])
        target = int(sequence[1])
        sequence[2] = float(sequence[2])
        if sequence[2]>max:
            sequence[2] = max
        c_adj[source, target] = sequence[2]
        c_adj[target, source] = sequence[2]
        line = f.readline()
    f.close()
    return c_adj

def get_noise_inputs():
    
    #Function to construct the noise input list of the generaive network's GCN units
    noise_inputs = []
    for i in range(window_size+1):
        noise_inputs.append(generative_noise(node, node))
    return noise_inputs

def generative_network(noise_input_phs, gcn_fact_phs):
    
    #Function to define the generative network
    
    result_gcn = [] 
    for i in range(window_size+1):
        noise = noise_input_phs[i]
        gcn_fact = gcn_fact_phs[i]
        gcn_wei = gcn_weis[i]
        gcn_conv = tf.matmul(gcn_fact, noise)
        r_gcn = tf.sigmoid(tf.matmul(gcn_conv, gcn_wei))
        r_gcn = tf.reshape(r_gcn, [1, node*gen_hid_num0])
        result_gcn.append(r_gcn)
    
    LSTM_cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(node*gen_hid_num0)]) 
    with tf.variable_scope("generative_network") as generative_network:
        LSTM_outputs, states = rnn.static_rnn(LSTM_cells, result_gcn, dtype=tf.float32)
        parameters_lstm = [var for var in tf.global_variables() if var.name.stargettswith(generative_network.name)]

    gen_output = tf.nn.sigmoid(tf.matmul(LSTM_outputs[-1], gen_output_wei) + gen_output_bias)

    return gen_output, parameters_lstm

def generative_noise(m, n):
    #Function to generative noises with uniform discribution
    return np.random.uniform(0, 1., size=[m, n])
    #return np.random.normal(0.5, 1, [m, n])

def get_KL(adj_est, gnd):
    sum_est = 0
    for r in range(node):
        for c in range(node):
            sum_est += adj_est[r, c]
    sum_gnd = 0
    for r in range(node):
        for c in range(node):
            sum_gnd += gnd[r, c]
    p = gnd/sum_gnd
    q = adj_est/sum_est
    edge_wei_KL = 0
    for r in range(node):
        for c in range(node):
            cur_KL = 0
            if q[r, c]>0 and p[r, c]>0:
                cur_KL = p[r, c]*np.log(p[r, c]/q[r, c])
            edge_wei_KL += cur_KL

    return edge_wei_KL

def mis_rate(adj_est, gnd):
    mis_sum = 0
    for r in range(node):
        for c in range(node):
            if (adj_est[r, c]>0 and gnd[r, c]==0) or (adj_est[r, c]==0 and gnd[r, c]>0):
                mis_sum += 1
    mis_rate = mis_sum/(node*node)

    return mis_rate

def disc_net(disc_input):

    #Function to define the discriminative network
    # Input layer -> hidden layer #1
    disc_h1 = tf.nn.sigmoid(tf.matmul(disc_input, disc_wei1) + disc_bias1)
    # Hidden layer #1 -> Output layer
    disc_logit = tf.matmul(disc_h1, disc_wei2) + disc_bias2
    disc_output = tf.nn.sigmoid(disc_logit)

    return disc_output, disc_logit

node = 38
time_num = 1000
window_size = 10
name = r"C:\Users\DELL\Desktop\GWA-T-13_Materna-Workload-Traces\Materna-Trace-2\01.csv"

max = 2000

pre_epoch_num = 1000
epoch_num = 4000
gen_hid_num0 = 1
gen_hid_num1 = 64

gcn_weis = []
for i in range(window_size+1):
    gcn_weis.append(tf.Variable(v_initialization(node, gen_hid_num0)))
gen_output_wei = tf.Variable(v_initialization(node*gen_hid_num0, node*node))
gen_output_bias = tf.Variable(tf.zeros(shape=[node*node]))
gen_output_params = [gen_output_wei, gen_output_bias]
disc_hid_num = 1024
disc_wei1 = tf.Variable(v_initialization(node*node, disc_hid_num))
disc_bias1 = tf.Variable(tf.zeros([disc_hid_num]))
disc_wei2 = tf.Variable(v_initialization(disc_hid_num, 1))
disc_bias2 = tf.Variable(tf.zeros([1]))
disc_params = [disc_wei1, disc_bias1, disc_wei2, disc_bias2]
clip_ops = []
for var in disc_params:
    clip_bound = [-0.01, 0.01]
    clip_ops.append(
        tf.assign(var, tf.clip_by_value(var, clip_bound[0], clip_bound[1]))
    )
clip_disc_wei = tf.group(*clip_ops)
gcn_fact_phs = []
noise_input_phs = []
for i in range(window_size+1):
    gcn_fact_phs.append(tf.placeholder(tf.float32, shape=[node, node]))
    noise_input_phs.append(tf.placeholder(tf.float32, shape=[node, node]))
gnd_ph = tf.placeholder(tf.float32, shape=(1, node*node))
gen_output, parameters_lstm = generative_network(noise_input_phs, gcn_fact_phs)
disc_real, disc_logit_real = disc_net(gnd_ph)
disc_fake, disc_logit_fake = disc_net(gen_output)

pre_gen_loss = tf.reduce_sum(tf.square(gnd_ph - gen_output))
pre_gen_opt = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(pre_gen_loss, var_list=(gcn_weis+parameters_lstm+gen_output_params))
gen_loss = -tf.reduce_mean(disc_logit_fake)
disc_loss = tf.reduce_mean(disc_logit_fake) - tf.reduce_mean(disc_logit_real)


disc_opt = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss, var_list=disc_params)
gen_opt = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss, var_list=(gcn_weis+parameters_lstm+gen_output_params))

avg_error = 0.0
avg_KL = 0.0
avg_mis = 0.0
cal_count = 0

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for t in range(window_size, time_num-2):

    gcn_facts = []
    for k in range(t-window_size, t+1):

        adj = get_data(name, k, node, max)/max
        gcn_fact = gcn_factor(adj)
        gcn_facts.append(gcn_fact)

    gnd = np.reshape(get_data(name, t+1, node, max ), (1, node*node))
    gnd /= max
    loss_list = []
    for epoch in range(pre_epoch_num):

        noise_inputs = get_noise_inputs()
        ph_dict = dict(zip(noise_input_phs, noise_inputs))
        ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        ph_dict.update({gnd_ph: gnd})
        _, pre_g_loss, pre_g_output = sess.run([pre_gen_opt, pre_gen_loss, gen_output], feed_dict=ph_dict)
        loss_list.append(pre_g_loss)
        if epoch%100==0:
            print('Pre-Train #%d, G-Loss: %f'%(epoch, pre_g_loss))
        if epoch>500 and loss_list[epoch]>loss_list[epoch-1] and loss_list[epoch-1]>loss_list[epoch-2]:
            break


    print('Train the GAN')
    for epoch in range(epoch_num):

        noise_inputs = get_noise_inputs()

        ph_dict = dict(zip(noise_input_phs, noise_inputs))
        ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        ph_dict.update({gnd_ph : gnd})
        _, d_loss = sess.run([disc_opt, disc_loss], feed_dict=ph_dict)

        noise_inputs = get_noise_inputs()

        ph_dict = dict(zip(noise_input_phs, noise_inputs))
        ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
        _, g_loss, g_output = sess.run([gen_opt, gen_loss, gen_output], feed_dict=ph_dict)
        _ = sess.run(clip_disc_wei)
        if epoch%100==0:
            print('GAN-Train #%d, D-Loss: %f, G-Loss: %f'%(epoch, d_loss, g_loss))

        gcn_facts = []
    for k in range(t-window_size+1, t+2):
        adj = get_data(name, k, node, max)/max
        gcn_fact = gcn_factor(adj)
        gcn_facts.append(gcn_fact)
    noise_inputs = get_noise_inputs()
    ph_dict = dict(zip(noise_input_phs, noise_inputs))
    ph_dict.update(dict(zip(gcn_fact_phs, gcn_facts)))
    output = sess.run([gen_output], feed_dict=ph_dict)
    adj_est = np.reshape(output[0]*max, (node, node))
    adj_est = (adj_est+adj_est.T)/2
    for r in range(node):
        adj_est[r, r] = 0
    for r in range(node):
        for c in range(node):
            if adj_est[r, c]<0.01:
                adj_est[r, c] = 0

    gnd = get_data(name, t+2, node, max)


    print('adj_est')
    for r in range(50):
        print('%.2f' % (adj_est[1, r]), end=' ')
    print()
    print('gnd')
    for r in range(50):
        print('%.2f' % (gnd[1, c]), end=' ')
    print()

    error = np.linalg.norm(gnd-adj_est, ord='fro')/(node*node)
    avg_error += error
    print('#%d Error: %f' % (t+2, error))

    edge_wei_KL = get_KL(adj_est, gnd)
    avg_KL += edge_wei_KL
    print('#%d Edge Weight KL: %f' % (t + 2, edge_wei_KL))

    mis_rate = mis_rate(adj_est, gnd)
    avg_mis += mis_rate
    print('#%d Mismatch Rate: %f' % (t + 2, mis_rate))

    print()

    cal_count += 1

    f = open("+UCSB-LSTM_GAN_GCN-rror.txt", 'a+')
    f.write('%d %f' % (t + 2, error))
    f.write('\n')
    f.close()

    f = open("+UCSB-LSTM_GAN_GCN-KL.txt", 'a+')

    f.write('%d %f' % (t + 2, edge_wei_KL))
    f.write('\n')
    f.close()

    f = open("+UCSB-LSTM_GAN_GCN-mis.txt", 'a+')

    f.write('%d %f' % (t + 2, mis_rate))
    f.write('\n')
    f.close()
avg_error /= cal_count
avg_KL /= cal_count
avg_mis /= cal_count

f = open("+UCSB-LSTM_GAN_GCN-rror.txt", 'a+')

f.write('Avg. Error %f' % (avg_error))
f.write('\n')
f.close()

f = open("+UCSB-LSTM_GAN_GCN-KL.txt", 'a+')

f.write('Avg. KL %f' % (avg_KL))
f.write('\n')
f.close()
f = open("+UCSB-LSTM_GAN_GCN-mis.txt", 'a+')

f.write('Avg. Mis %f' % (avg_mis))
f.write('\n')
f.close()
