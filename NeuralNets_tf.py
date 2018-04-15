import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('tmp/data',one_hot=True)

n_h1=500
n_h2=500
n_h3=500

n_class=10
batch_s=100

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')


def neural_net(data):
 	hidden_1={'w':tf.Variable(tf.random_normal([784,n_h1])),
 				'b':tf.Variable(tf.random_normal([n_h1]))} 
 	
 	hidden_2={'w':tf.Variable(tf.random_normal([n_h1,n_h2])),
 				'b':tf.Variable(tf.random_normal([n_h2]))} 
 	
 	hidden_3={'w':tf.Variable(tf.random_normal([n_h2,n_h3])),
 				'b':tf.Variable(tf.random_normal([n_h3]))}
 	
 	output={'w':tf.Variable(tf.random_normal([n_h3,10])),
 				'b':tf.Variable(tf.random_normal([10]))}

 	l1=tf.add(tf.matmul(data,hidden_1['w']),hidden_1['b'])
 	l1=tf.nn.relu(l1)
 	
 	l2=tf.add(tf.matmul(l1,hidden_2['w']),hidden_2['b'])
 	l2=tf.nn.relu(l2)

 	l3=tf.add(tf.matmul(l2,hidden_3['w']),hidden_3['b'])
 	l3=tf.nn.relu(l3)

 	out=tf.matmul(l3,output['w'])+output['b']

 	return out


def train_nn(x):
	prediction=neural_net(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

	optimizer=tf.train.AdamOptimizer().minimize(cost)

	epochs=5

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for ep in range(epochs):
			epoch_loss=0 	

			for _ in range(int(mnist.train.num_examples/batch_s)):

				e_x,e_y=mnist.train.next_batch(batch_s)

				_,c=sess.run([optimizer,cost],feed_dict={x:e_x,y:e_y})
				epoch_loss+=c

			print('Epoch',ep,'completed out of',epochs,'loss:',epoch_loss)
				
		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print('accuracy',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))	


# train_nn(x)

print(int(mnist.train.num_examples))
