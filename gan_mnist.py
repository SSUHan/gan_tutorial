import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import datetime

from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('MNIST_data/')

# sample_image = mnist_data.train.next_batch(1)[0]
# print(sample_image.shape)
# sample_image = sample_image.reshape([28,28])
# plt.imshow(sample_image, cmap='gray')
# plt.show()

def discriminator(images, reuse=False):
	with tf.variable_scope(tf.get_variable_scope(), reuse=reuse) as scope:
		# First Convolutional and pool layers
		# This finds 32 different 5*5 features
		d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
		d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1,1,1,1], padding='SAME')
		d1 = d1 + d_b1
		d1 = tf.nn.relu(d1)
		d1 = tf.nn.avg_pool(d1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		# Second Convolutional layer
		d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1,1,1,1], padding='SAME')
		d2 = d2 + d_b2
		d2 = tf.nn.relu(d2)
		d2 = tf.nn.avg_pool(d2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

		# First Fully Connected layer
		d_w3 = tf.get_variable('d_w3', [7*7*64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable('d_b3', [1024], initializer=tf.constant_initializer(0))
		d3 = tf.reshape(d2, [-1, 7*7*64]) # Flatten
		d3 = tf.matmul(d3, d_w3)
		d3 = d3 + d_b3
		d3 = tf.nn.relu(d3)

		# Last Output layer
		d_w4 = tf.get_variable('d_w4', [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
		d4_logit = tf.matmul(d3, d_w4) + d_b4
		d4_prob = tf.nn.sigmoid(d4_logit)

		return d4_prob, d4_logit

def generator(z, batch_size, z_dim):
	g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02)) # 3136 = 28*28*2*2
	g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g1 = tf.matmul(z, g_w1) + g_b1
	g1 = tf.reshape(g1, [-1, 56, 56, 1])
	g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
	g1 = tf.nn.relu(g1)

	# Generate 50 features
	g_w2 = tf.get_variable('g_w2', [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b2 = tf.get_variable('g_b2', [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
	g2 = g2 + g_b2
	g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
	g2 = tf.nn.relu(g2)
	g2 = tf.image.resize_images(g2, [56, 56])

	# Generate 25 features
	g_w3 = tf.get_variable('g_w3', [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b3 = tf.get_variable('g_b3', [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g3 = tf.nn.conv2d(g2, g_w3, strides=[1,2,2,1], padding='SAME')
	g3 = g3 + g_b3
	g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
	g3 = tf.nn.relu(g3)
	g3 = tf.image.resize_images(g3, [56, 56])

	# Final convolution with one output channel
	g_w4 = tf.get_variable('g_w4', [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
	g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
	g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
	g4 = g4 + g_b4
	g4 = tf.sigmoid(g4)

	# Dimension of g4: batch_size * 28 * 28 * 1
	return g4

z_dimensions = 100 
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

generated_image_output = generator(z_placeholder, 1, z_dimensions)
z_batch = np.random.normal(0, 1, [1, z_dimensions])

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	generated_image = sess.run(generated_image_output, feed_dict={z_placeholder:z_batch})
# 	generated_image = generated_image.reshape([28, 28])
# 	plt.imshow(generated_image, cmap='gray')
# 	plt.show()

tf.reset_default_graph()
batch_size = 50

# z_placeholder is for feeding input noise to the "generator"
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# x_placeholder is for feeding input images to the "discriminator"
x_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x_placeholder')

# Gz holds the generated images
Gz = generator(z_placeholder, batch_size, z_dimensions)

# Dx will hold discriminator prediction probabilities
# for the real MNIST images
Dx_prob, Dx_logit = discriminator(x_placeholder)

# Dg will hold discriminator prediction probabilities for generated images
Dg_prob, Dg_logit = discriminator(Gz, reuse=True)

# Compare real images loss probabilities : 1 <-> Dx output
# tf.ones_like : Make same shape of 1 matrix
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx_logit, labels=tf.ones_like(Dx_logit)))

# Compare fake images loss probabilities : 0 <-> Dg outut
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logit, labels=tf.zeros_like(Dg_logit)))

d_loss = d_loss_fake + d_loss_real

# generator wants the discriminator to output a value close to 1 when it's given an image from the generator
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg_logit, labels=tf.ones_like(Dg_logit)))

tvars = tf.trainable_variables()
d_tvars = [var for var in tvars if 'd_' in var.name]
g_tvars = [var for var in tvars if 'g_' in var.name]

print(d_tvars)
print(g_tvars)

d_trainer = tf.train.AdamOptimizer(0.0003).minimize(d_loss, var_list=d_tvars)
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_tvars)

tf.get_variable_scope().reuse_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)

images_for_tensorboard = generator(z_placeholder, batch_size, z_dimensions)
tf.summary.image('Generated_Image', images_for_tensorboard, 5)
merged = tf.summary.merge_all()

logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
writer = tf.summary.FileWriter(logdir, sess.graph)


for i in range(300):
	z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
	real_image_batch = mnist_data.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
	_, _, dLoss, gLoss  = sess.run([d_trainer, g_trainer, d_loss, g_loss], 
									feed_dict={z_placeholder:z_batch, x_placeholder:real_image_batch})
	if i % 100 == 0:
		print("Discriminator loss : {}, Generator loss : {}".format(dLoss, gLoss))


saver = tf.train.Saver()

for i in range(100000):
	# train discriminator
	z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
	real_image_batch = mnist_data.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
	_, _, dLoss, gLoss = sess.run([d_trainer, g_trainer, d_loss, g_loss], 
	                                      feed_dict={x_placeholder:real_image_batch, z_placeholder:z_batch})

	# train generator
	z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
	_ = sess.run(g_trainer, feed_dict={z_placeholder:z_batch})

	if i % 10 == 0:
		# Update tensorboard
		z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])
		summary = sess.run(merged, feed_dict={z_placeholder:z_batch, x_placeholder:real_image_batch})
		writer.add_summary(summary, i)
	    
	if i % 100 == 0:
		# Every 100 iters, show generated images
		print("i : {}, at : {}".format(i, datetime.datetime.now()))
		z_batch = np.random.normal(0, 1, [1, z_dimensions])
		generating_image_tensor = generator(z_placeholder, 1, z_dimensions)
		generated_image = sess.run(generating_image_tensor, feed_dict={z_placeholder:z_batch})
		# plt.imshow(generated_image[0].reshape([28,28]), cmap='gray')
		# plt.show()
	    
		# Show discriminator's estimate
		im = generated_image[0].reshape([1, 28, 28, 1])
		discriminator_result_tensor, _ = discriminator(x_placeholder)
		estimate_result = sess.run(discriminator_result_tensor, feed_dict={x_placeholder:im})
		print("Estimated result : {}".format(estimate_result))

	if i % 10000 == 0:
		# Save weights
		saver.save(sess, 'pretrained-model/lossum_gan.ckpt', global_step=i)
