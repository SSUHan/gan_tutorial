import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
import datetime

from tensorflow.examples.tutorials.mnist import input_data

from gan_mnist_ops import discriminator, generator

mnist_data = input_data.read_data_sets('MNIST_data/')

# sample_image = mnist_data.train.next_batch(1)[0]
# print(sample_image.shape)
# sample_image = sample_image.reshape([28,28])
# plt.imshow(sample_image, cmap='gray')
# plt.show()


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
